#!/usr/bin/python3

import numpy as np
from tensorflow import keras
# from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import matplotlib.pyplot as plt

### Partie A - Création des données

n = 200  # pour le nb de points dans la grille
xmin, xmax, ymin, ymax = -2.05, 2.05, -0.75, 1.05

# Fonction f : lemniscate de Bernouilli + ellipse
def f(x,y):
    res = ((x**2+y**2)**2 - 4*(x**2-y**2))*((x-1/2)**2 + 4*(y-1/3)**2-2)
    # res = ((x**2+y**2)**2 - 4*(x**2-y**2))
    return  res


# Generations des points rouges/bleus
# Rouge si f <= 0, bleu sinon
def donnees(n):
    VX = np.linspace(xmin, xmax, n)
    VY = np.linspace(ymin, ymax, n)
    X, Y = np.meshgrid(VX, VY)
    Z = f(X, Y)
    Zbool = Z <= 0
    Zint = Zbool.astype(np.int)
    return X, Y, Zint


 # Test   
X, Y, Z = donnees(n)
# print(X,Y,Z)
plus = np.sum(Z)
moins = np.size(Z)-plus
print("Points positifs :",plus)
print("Points négatifs :",moins)


def donnees_keras(n):
    X, Y, Z = donnees(n)
    X = X.flatten()
    Y = Y.flatten()

    # Z = Z.flatten()
    liste_points = []
    for i in range(n**2):
        points = (X[i], Y[i])
        liste_points.append(points)

    entree = np.array(liste_points)
    # print(entree)
    print(entree.shape)

    sortie = Z.reshape((n**2,1))

    # print(sortie)
    print(sortie.shape)
    return entree, sortie

# Test
donnees_keras(n)

# Affichage
def graphique_donnees(X,Y,Z):

    # for x, y in carres_rouges:    # points
    #     plt.scatter(x, y, marker='s', color='red')
    # for x, y in ronds_bleus:    # points
    #     plt.scatter(x, y, color='blue')   

    plt.scatter(X, Y, c=Z, s=2, cmap='bwr')

    plt.axis('equal')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.tight_layout()
    # plt.savefig('retro_05_p=5.png')
    plt.show()
    return

# Test 
# graphique_donnees(X,Y,Z)


### Partie B - Réseau de neurones

modele = Sequential()

p = 5
q = p
r = p

# 2 entrées (x,y), 
# première couche : p neurones
# deuxième couche : q neurones
# troisième couche (sortie) : 1 neurone 
# activation = à voir

# Première couche : p neurones (entrée de dimension 2)
modele.add(Dense(p, input_dim=2, activation='relu'))

# Deuxième couche : q neurones
modele.add(Dense(q, activation='relu'))

# Troisième couche : r neurones
modele.add(Dense(r, activation='relu'))

# Couche de sortie : 1 neurone
modele.add(Dense(1, activation='relu'))

# Descente de gradient
# mysgd = optimizers.SGD(lr=0.1)
# modele.compile(loss='mean_squared_error', optimizer=mysgd)
modele.compile(loss='mean_squared_error', optimizer='adam')

# Données
X_train, Y_train = donnees_keras(n)

# Calcul des poids
# modele.fit(X_train, Y_train, epochs=10000, batch_size=len(X_train), verbose = 1)
modele.fit(X_train, Y_train, epochs=10, batch_size=100, verbose = 1)

print(modele.summary())

### Partie C - Résultats

Zpredict = modele.predict(X_train)
Zpredict = Zpredict.reshape((n,n))
Zbool = Zpredict >= 0.5
Zint = Zbool.astype(np.int)
# print(Z)
# print(Zpredict)
graphique_donnees(X, Y, Zint)

