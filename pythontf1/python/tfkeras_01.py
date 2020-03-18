#!/usr/bin/python3

# import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Architecture du réseau
modele = Sequential()

# Couches de neurones
modele.add(Dense(2, input_dim=1, activation='relu'))
modele.add(Dense(1, activation='relu'))


# Couche 0 - Définir à la main les poids
coeff = np.array([[1.,-0.5]])
biais = np.array([-1,1])
poids = [coeff,biais]
modele.layers[0].set_weights(poids)

# Vérification
verif_poids = modele.layers[0].get_weights()
print(verif_poids)

# Couche 1 - Définir à la main les poids
coeff = np.array([[1.0],[1.0]])
biais = np.array([0])
poids = [coeff,biais]
modele.layers[1].set_weights(poids)

# Vérification
verif_poids = modele.layers[1].get_weights()
print(verif_poids)


# Entrée/Sortie : une seule valeur
entree = np.array([[3.0]])
sortie = modele.predict(entree)
print('Entrée :',entree,'Sortie :',sortie)

# print(np.linspace(0,1,3))
# # Entrée/Sortie : une liste de valeurs
# entree = np.array([[1.0], [2.0], [3.0]])
# # entree = np.linspace(0,1,3)
# entree = [1,2,3]
# sortie = modele.predict(entree)
# print('Entrée :',entree)
# print('Sortie :',sortie)


# Affichage graphique
import matplotlib.pyplot as plt

liste_x = np.linspace(-2, 3, num=100)
entree =  np.array([[x] for x in liste_x])
sortie = modele.predict(entree)
liste_y = np.array([y[0] for y in sortie])
# print(liste_x)
# print(liste_y)
plt.plot(liste_x,liste_y)
plt.tight_layout()
# plt.savefig('pythontf-keras-01.png')
plt.show()


