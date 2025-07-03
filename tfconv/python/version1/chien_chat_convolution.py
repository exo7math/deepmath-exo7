# F. Chollet https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d
# Kaggle https://www.kaggle.com/c/dogs-vs-cats/data
# https://pythonistaplanet.com/image-classification-using-deep-learning/


import numpy as np
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import load_model

# Partie A. Données

# A télécharger sur https://www.kaggle.com/c/dogs-vs-cats/data
# après inscription

train_directory = '/home/arnaud/Exo7/deepmath/chiens-vs-chats/donnees/train'
test_directory = '/home/arnaud/Exo7/deepmath/chiens-vs-chats/donnees/test'
image_width = 64
image_height = 64
nb_train_images = 20000
nb_test_images = 5000

# Transforme les images en données d'apprentissage : 
# (a) reformate les images en taille unique, 
# (b) créé une classification (à partir de chaque sous-répertoire) 0 pour les chats  et (1) pour les chiens 

train_datagen = ImageDataGenerator(rescale =1./255)
training_set = train_datagen.flow_from_directory(train_directory,
                                                target_size=(image_width,image_height),
                                                batch_size= 32,
                                                shuffle=True, seed=13,
                                                class_mode='binary')

# Idem pour les données de test 
test_datagen = ImageDataGenerator(rescale =1./255)
test_set = test_datagen.flow_from_directory(test_directory,
                                           target_size = (image_width,image_width),
                                           batch_size = 32,
                                           shuffle=True, seed=13,
                                           class_mode ='binary')


# Partie A bis. Afficher des images

# A refaire
import matplotlib.pyplot as plt

def affiche_images():
    plt.axis('off')
    X, Y = training_set.next()  # a batch of 32 images
    # X = images, Y = categories chat/chien
    for i in range(9):
        image = X[i]
        plt.subplot(330 + 1 + i)
        if Y[i] == 0:
            animal = 'Chat'
        else:
            animal = 'Chien'
        plt.title(animal)
        plt.imshow(image, interpolation='nearest')
    plt.tight_layout()
    # plt.savefig('tfconv-chienchat-train.png')
    plt.show()

    return

# affiche_images()

# Partie B. Réseau 

modele = Sequential()

# Première couche de convolution : 32 neurones, convolution 3x3, activation relu
modele.add(Conv2D(64, kernel_size=3, padding='same', activation='relu', input_shape=(image_width,image_height,3)))

# Mise en commun (pooling)
modele.add(MaxPooling2D(pool_size=(2, 2)))

# Dropout
modele.add(Dropout(0.5))

# Deuxième couche de convolution : 32 neurones
modele.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))

# Mise en commun (pooling)
modele.add(MaxPooling2D(pool_size=(2, 2)))

# Dropout
modele.add(Dropout(0.5))

# Troisième couche de convolution : 32 neurones
modele.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))

# Mise en commun (pooling)
modele.add(MaxPooling2D(pool_size=(2, 2)))

# Dropout
modele.add(Dropout(0.5))

# Aplatissage 
modele.add(Flatten())

# Couche dense : 32 neurones
modele.add(Dense(32, activation='relu'))

# Couche de sortie : 1 neurone
modele.add(Dense(1, activation='sigmoid'))

# Méthode de gradient
modele.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Affiche un résumé
print(modele.summary())


# Partie C. Apprentissage

# Reprendre les poids déjà calculés
# modele.load_weights('weights_chien_chat.h5')

# modele.fit(X_train, Y_train, epochs=1, batch_size=32)
history = modele.fit_generator(training_set,
                        steps_per_epoch = nb_train_images // 32,
                        epochs = 1)

# Sauvegarde
modele.save_weights(filepath='weights_chien_chat.h5')


# Partie D. Résultats

score = modele.evaluate_generator(test_set, verbose=0)
print('Test erreur (loss) :', score[0])
print('Test précision (accuracy) :', score[1])

plt.plot(history.history['accuracy'])
plt.savefig('tfconv-chienchat-acc.png')
plt.show()


# 64x64 pixels

# c16-c32-c32-d32       79937 poids, 10 époques : 77%/78%
#                                    15 époques : 79%/80%
#                                    20 époques : 80%/80%
#                                    25 époques : 81%/80%      

# c64-c64-c64-d32       206 785 poids,  5 époques : 75%/78%
#                                      10 époques : 80%/81%  
#                                      15 époques : 82%/83%  
#                                      20 époques : 83.7%/83.6%  
#                                      25 époques : 84.7%/83.6%  
#                                      30 époques : 85.9%/84.3%  
#                                      35 époques : 86.9%/85.4% 
#                                      40 époques : 87.2%/85.8%  


# Partie E. Visualisation

test_set = test_datagen.flow_from_directory(test_directory,
                                           target_size = (image_width,image_width),
                                           batch_size = 32,
                                           shuffle=True, seed=14,
                                           class_mode ='binary')

def affiche_images_test():
    plt.axis('off')
    X, Y = test_set.next()  # a test batch 
    # X = images, Y = categories chat/chien
    Y_predict = modele.predict(X)    
    for i in range(9):
        image = X[i]
        plt.subplot(330 + 1 + i)
        if Y[i] == 0:
            animal = 'Attendu : chat\n'
        else:
            animal = 'Attendu : chien\n'
        perc_chien = int(round(100*np.max(Y_predict[i])))
        perc_chat = 100 - perc_chien
        if Y_predict[i] <= 0.5:
            animal += 'Prédit : chat (%d%%)' % perc_chat
        else:
            animal += 'Prédit : chien (%d%%)' % perc_chien
        plt.title(animal)
        plt.imshow(image, interpolation='nearest')
    plt.tight_layout()
    plt.savefig('tfconv-chienchat-test.png')
    plt.show()

    return

affiche_images_test()
