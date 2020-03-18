#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


# Récupération des données
(X_train_data, Y_train_data), (X_test_data, Y_test_data) = mnist.load_data()


# Affichage d'un chiffre (données d'apprentissage)
def affiche_chiffre_train(i):
    plt.imshow(X_train_data[i], cmap='Greys')
    plt.title('Attendu %d' % Y_train_data[i], fontsize=30)
    plt.tight_layout()
    # plt.savefig('tf2-chiffre-train-%d.png' %i)
    plt.show()
    return

# for i in range(10):
#     affiche_chiffre_train(i)


# Taille des données
print("Données d'apprentissage brutes X", X_train_data.shape)
print("Données d'apprentissage brutes Y", Y_train_data.shape)


N = X_train_data.shape[0]  # 60 000 données
print("Nombre de  données d'apprentissage :", N)
print("Taille d'une image :", X_train_data[0].shape)
print("Première image :\n", X_train_data[0])

# Mise sous forme de vecteur
X_train = np.reshape(X_train_data,(N,784))

# Normalisation (on se ramène entre 0 et 1)
X_train = X_train/255

print("Taille d'un vecteur :", X_train[0].shape)
print("Premièr vecteur image :\n", X_train[0])

# Résultat attendu

Y_train = to_categorical(Y_train_data, num_classes=10)

print("Résultat brut attendu pour la premier chiffre :", Y_train_data[0])
print("Résultat sous forme de liste :", Y_train[0])



# Pour le cours

# Affichage d'un chiffre (données de test)
def affiche_chiffre_test(i):
    plt.imshow(X_test_data[i], cmap='Greys')
    # plt.title('Attendu %d' % Y_test_data[i])
    plt.tight_layout()
    # plt.savefig('tf2-chiffre-test-%d.png' %i)
    plt.show()
    return

# for i in range(10):
    # affiche_chiffre_test(i)