# Inspired from :
# https://www.kaggle.com/drscarlat/imdb-sentiment-analysis-keras-and-tensorflow
# https://raw.githubusercontent.com/keras-team/keras/master/examples/imdb_lstm.py

import numpy as np
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Partie A. Données

from tensorflow.keras.datasets import imdb

nb_mots_total = 1000   # On ne garde que les n=1000 mots les plus fréquents 
nb_mots_texte = 50     # Pour chaque critique on ne garde que 50 mots 
(X_train_data, Y_train), (X_test_data, Y_test) = imdb.load_data(num_words=nb_mots_total)


# Données sous forme de vecteurs

def vectorisation_critiques(X_data):
    vecteurs = np.zeros((len(X_data), nb_mots_total))
    for i in range(len(X_data)):
        for c in X_data[i]:
            vecteurs[i,c] = 1.0
    return vecteurs

X_train = vectorisation_critiques(X_train_data)
X_test = vectorisation_critiques(X_test_data)

# Partie B. Réseau 

modele = Sequential()

p = 5
modele.add(Dense(p, input_dim=nb_mots_total, activation='relu'))
modele.add(Dense(p, activation='relu'))
modele.add(Dense(p, activation='relu'))
modele.add(Dense(1, activation='sigmoid'))

# mysgd = optimizers.SGD(lr = 0.1, decay=1e-6, momentum=0.9, nesterov=True)
modele.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

print(modele.summary())


# Partie C. Apprentissage
list_loss_train = []
list_acc_train = []
list_loss_test = []
list_acc_test = []

for k in range(30):   # Nb d'époques à la main 
    # Apprentissage
    history = modele.fit(X_train, Y_train, epochs=1, batch_size=32, verbose=2)
    list_loss_train.append(history.history['loss'][0])
    list_acc_train.append(history.history['accuracy'][0])

    # Validation sur les données de test
    score = modele.evaluate(X_test, Y_test, verbose=0)
    list_loss_test.append(score[0])
    list_acc_test.append(score[1])    

# Affichage des erreurs
import matplotlib.pyplot as plt

plt.plot(list_acc_train, label="précision sur données d'apprentissage")
plt.plot(list_acc_test, label="précision sur données de test")
plt.xlabel('époque')
plt.ylabel('précision')
plt.legend()
plt.tight_layout()
# plt.savefig('tf2-texte-overfit-acc.png')
plt.show()


plt.plot(list_loss_train, label="erreur sur données d'apprentissage")
plt.plot(list_loss_test, label="erreur sur données de test")
plt.xlabel('époque')
plt.ylabel('erreur')
plt.legend()
plt.tight_layout()
# plt.savefig('tf2-texte-overfit-loss.png')
plt.show()