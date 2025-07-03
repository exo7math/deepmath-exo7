#!/usr/bin/python3

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Architecture du réseau
modele = Sequential()

# Couches de neurones
modele.add(Dense(3, input_dim=2, activation='sigmoid'))
modele.add(Dense(1, activation='sigmoid'))

# Couche 0 - Définir à la main les poids
coeff = np.array([[1.0,3.0,-5.0],[2.0,-4.0,-6.0]])
biais = np.array([-1.0,0.0,1.0])
poids = [coeff,biais]
modele.layers[0].set_weights(poids)

# Vérification
verif_poids = modele.layers[0].get_weights()
print(verif_poids)

# Couche 1 - Définir à la main les poids
coeff = np.array([[1.0],[1.0],[1.0]])
biais = np.array([-3.0])
poids = [coeff,biais]
modele.layers[1].set_weights(poids)

# Vérification
verif_poids = modele.layers[1].get_weights()
print(verif_poids)


# Entrée/Sortie : une seule valeur
entree = np.array([[7,-5]])
sortie = modele.predict(entree)
print('Entrée :',entree,'Sortie :',sortie)
print(np.shape(sortie))


# Affichage graphique 3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

VX = np.linspace(-5, 5, 30)
VY = np.linspace(-5, 5, 30)
X,Y = np.meshgrid(VX, VY)
entree = np.c_[X.ravel(), Y.ravel()]


sortie = modele.predict(entree)
print(np.shape(sortie))

Z = sortie.reshape(X.shape)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('axe x')
ax.set_ylabel('axe y')
ax.set_zlabel('axe z')

ax.plot_surface(X, Y, Z)

# ax.view_init(40, 120)
# ax.view_init(60, 125)
# plt.tight_layout()
# plt.savefig('pythontf-keras-02b.png')
plt.show()