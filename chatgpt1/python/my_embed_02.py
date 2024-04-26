import nltk
from nltk.corpus import reuters
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout



# Où sont les données ?
# Il faut d'abord avoir exécuter my_embed_01.py

vocab = np.load("vocab.npy")
vocab = list(vocab)
N = len(vocab)
proba = np.load("proba.npy")

# Vérifications
print("Nombre de mots du vocabulaires ", N) 
print("Début du vocabulaire : ", vocab[0:10])
print("Fin du vocabulaire : ", vocab[-10:])
print("Shape de proba : ", proba.shape)


### Partie A - Création des données

# En entrée tout le vocabulaire. 
# Un mot du vocab est représenté par un vecteur de taille N avec un 1 à la position du mot et des 0 ailleurs
X_train = np.identity(N)
# X_train = np.reshape(X_train, (N,N,1))
print(X_train.shape)
# print(X_train[0])

# En sortie la liste des vecteurs de probabilités
Y_train = proba


### Partie B - Réseau de neurones
embed_dim = 50
modele = Sequential()

# modele.add(Conv1D(8, 3, activation='relu',input_shape=(N,1)))
# modele.add(MaxPooling1D(2))
# modele.add(Conv1D(8, 3, activation='relu'))
# modele.add(MaxPooling1D(2))

modele.add(Dense(50, input_dim=N, activation='relu'))
modele.add(Dense(50, activation='relu'))
modele.add(Dense(embed_dim, activation='relu'))
modele.add(Flatten())

# Couche de sortie : N neurones
modele.add(Dense(N, activation='softmax'))

# Descente de gradient
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.0, nesterov=True)
modele.compile(loss='mean_squared_error',
               #loss='categorical_crossentropy',
              optimizer='adam',  
              metrics=['accuracy'])

print(modele.summary())


# Calcul des poids
modele.fit(X_train, Y_train, batch_size=32, epochs=1000, verbose=1)


### Partie C - Résultats

score = modele.evaluate(X_train, Y_train, verbose=1)
print('Test loss:', score)
# print('Test accuracy:', score[1])


# Dernière couche avant softmax et la sortie
# Fonction qui prend en entrée une image et renvoie la sortie de la couche (dans un espace de dimension nb_neurones)
func = K.function([modele.input], [modele.layers[-2].output])


### Partie D - Visualisation

# Analyse en composantes principales
# cad meilleure projection de l'espace des vecteurs sur un espace de dimension 2
# https://stackoverflow.com/questions/36771525/python-pca-projection-into-lower-dimensional-space
def pca(X, k=2):
    # Centrer les données
    Xmean = X.mean(axis=0)
    XX = X - Xmean
    # Matrice de covariance
    C = np.dot(XX.T, XX) / (XX.shape[0] - 1)
    # Décomposition en valeurs propres
    d, u = np.linalg.eigh(C)
    # Tri des valeurs propres
    idx = np.argsort(d)[::-1]
    # Tri des vecteurs propres
    u = u[:, idx]
    # Projection sur les k premiers vecteurs propres
    U = u[:, :k]
    return np.dot(XX, U), Xmean, u[:, :k], d[idx]



def print_vocab():
    """ Affichage du vocabulaire """

    # Affichage des mots
    for i in range(0, N, 10):
        print(vocab[i:i+10])

    return

print_vocab()


def plongements_mots():
    """ Visualisation des plongements de mots """

    # Liste de mots par catégories
    liste_mots = []
    liste_mots += ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august']
    liste_mots += ['september', 'october', 'november', 'december']
    # liste_mots += ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    # liste_mots += ['europe', 'asia', 'america', 'africa', 'australia']
    # liste_mots += ['london', 'paris', 'berlin', 'rome', 'madrid', 'moscow']
    liste_mots += ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
    # liste_mots += ['company', 'corporation', 'firm', 'business', 'enterprise', 'industry']
    # liste_mots += ['dollars', 'euros', 'yens', 'pounds', 'francs']
    # liste_mots += ['sales', 'turnover', 'revenue', 'income', 'profit', 'deficit']
    # liste_mots += ['president', 'chairman', 'director', 'manager', 'ceo', 'cfo', 'boss']
    # liste_mots += ['bank', 'insurance', 'broker', 'fund', 'investment', 'trading', 'exchange']
    # liste_mots += ['government', 'parliament', 'minister', 'senator', 'congress', 'president']
    # liste_mots += ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'brown', 'black', 'white']
    # liste_mots += ['european', 'british', 'french', 'english', 'spanish', 'italian', 'russian', 'chinese', 'japanese', 'canadian', 'american']
    # liste_mots += ['will', 'are', 'was', 'were', 'have', 'has', 'had', 'do', 'does', 'did']
    # liste_mots += ['price', 'prices', 'cost', 'value', 'sales', 'share', 'shares', 'trade', 'rates']

    # On ne garde que les mots du vocabulaire 
    liste_mots = [w for w in liste_mots if w in vocab]

    nbsamples = len(liste_mots)
    print("Nombre de mots à visualiser : ", nbsamples)
    print(liste_mots)

    indices = [vocab.index(w) for w in liste_mots]
    YY = func(X_train[indices,:])      # fonction récupérer via l'avant-dernière couche
    # print(type(YY[0]))

    # Affichage 2D
    Z, _, _, _ = pca(YY[0], k=2)

    for i in range(nbsamples):
        plt.scatter(Z[i, 0], Z[i, 1])   
        plt.text(Z[i, 0], Z[i, 1], vocab[indices[i]])
        
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    # plt.savefig("plongement-mots-1.png", dpi=600)
    plt.show()



    # Affichage 3D
    Z, _, _, _ = pca(YY[0], k=3)

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = plt.subplot(111, projection="3d")

    for i in range(nbsamples):
        ax.scatter(Z[i, 0], Z[i, 1], Z[i, 2])
        ax.text(Z[i, 0], Z[i, 1], Z[i, 2], vocab[indices[i]])

    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    ax.view_init(azim=-40, elev=20)

    # plt.show()

    return

plongements_mots()


