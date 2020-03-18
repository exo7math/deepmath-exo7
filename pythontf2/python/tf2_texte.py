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

nb_mots_total = 5000   # On ne garde que les n=1000 mots les plus fréquents 
nb_mots_texte = 50     # Pour chaque critique on ne garde que 50 mots 
(X_train_data, Y_train), (X_test_data, Y_test) = imdb.load_data(num_words=nb_mots_total)

# Partie A bis. Afficher d'un texte

# Afficher une critique et sa note 
def affiche_texte(num):
    index_mots = imdb.get_word_index()
    index_mots_inverse = dict([(value, key) for (key, value) in index_mots.items()])
    critique_mots = ' '.join([index_mots_inverse.get(i - 3, '??') for i in X_train_data[num]])
    print("Critique :\n", critique_mots)
    print("Note 0 (négatif) ou 1 (positif) ? :", Y_train[num])
    print("Critique (sous forme brute) :\n", X_train_data[num])
    return

affiche_texte(123)   # affichage de la critique numéro 123

# Partie A ter. Données sous forme de vecteurs

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

p = 10
modele.add(Dense(p, input_dim=nb_mots_total, activation='relu'))
modele.add(Dense(p, activation='relu'))
modele.add(Dense(p, activation='relu'))
modele.add(Dense(1, activation='sigmoid'))

modele.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

print(modele.summary())


# Partie C. Apprentissage
modele.fit(X_train, Y_train, epochs=10, batch_size=32)


# Partie D. Résultats

score = modele.evaluate(X_test, Y_test, verbose=0)
print('Test erreur (loss) :', score[0])
print('Test précision (accuracy) :', score[1])


Y_predict = modele.predict(X_test)


# Afficher une critique et sa note 
def affiche_texte_test(num):
    index_mots = imdb.get_word_index()
    index_mots_inverse = dict([(value, key) for (key, value) in index_mots.items()])
    critique_mots = ' '.join([index_mots_inverse.get(i - 3, '??') for i in X_test_data[num]])
    print("Critique :\n", critique_mots)
    print("Note attendue 0 (négatif) ou 1 (positif) ? :", Y_test[num])
    print("Note prédite 0 (négatif) ou 1 (positif) ? :", Y_predict[num][0])
    return

affiche_texte_test(111)   # prédiction pour la critique test numéro 111