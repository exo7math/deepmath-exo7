import numpy as np
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Partie A. Données

# A télécharger sur https://www.kaggle.com/c/dogs-vs-cats/data
train_directory = '/home/arnaud/Exo7/deepmath/chiens-vs-chats/donnees/train'
test_directory = '/home/arnaud/Exo7/deepmath/chiens-vs-chats/donnees/test'
image_width = 64
image_height = 64
nb_train_images = 20000

# Transforme les images en données d'apprentissage : 
# (a) reformate les images en taille unique, 
# (b) créé une classification (à partir de chaque sous-répertoire) 0 pour les chats  et (1) pour les chiens 
train_datagen = ImageDataGenerator(rescale =1./255)
training_set = train_datagen.flow_from_directory(train_directory,
                                                target_size=(image_width,image_height),
                                                batch_size= 32,
                                                shuffle=True, seed=13,
                                                class_mode='binary')

# Partie B. Réseau 
modele = Sequential()

modele.add(Conv2D(64, kernel_size=3, padding='same', activation='relu', 
                                     input_shape=(image_width,image_height,3)))
modele.add(MaxPooling2D(pool_size=(2, 2)))
modele.add(Dropout(0.5))

modele.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
modele.add(MaxPooling2D(pool_size=(2, 2)))
modele.add(Dropout(0.5))

modele.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
modele.add(MaxPooling2D(pool_size=(2, 2)))
modele.add(Dropout(0.5))

modele.add(Flatten())
modele.add(Dense(32, activation='relu'))
modele.add(Dense(1, activation='sigmoid'))

modele.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Partie C. Apprentissage
history = modele.fit_generator(training_set,
                        steps_per_epoch = nb_train_images // 32,
                        epochs = 10)
