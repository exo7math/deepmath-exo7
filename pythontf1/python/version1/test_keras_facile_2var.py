#!/usr/bin/python3


################################################
### Exemple d'utilisation du module keras_facile
### Cas de deux variables
################################################


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Importer le module
from keras_facile import *


### Exemple 1 : un seul neurone
def exemple1():
	print("\n--- Exemple 1 : un seul neurone ---")
	# Architecture du réseau
	modele = Sequential()
	modele.add(Dense(1, input_dim=2, activation='sigmoid', name='Couche_0'))

	# Définition des poids neurone par neurone
	definir_poids(modele,0,0,[1,2],-1)  # definir_poids(modele,couche,rang,coeff,biais)
	affiche_poids(modele,0)        # affiche poids de la couche 0

	# # Exemples d'évaluation
	entree = [2.0,1.0]
	# entree = np.array([2.0, 3.0])

	sortie = evaluation(modele,entree)
	print("\n=== Evaluation ===")
	print('Entrée :',entree)
	print('Sortie :',sortie)

	# Affichage graphique de la fonction définie par le réseau sur [a,b] (ici [-5,5])
	affichage_evaluation_deux_var_3d(modele,-5,5,-5,5)
	affichage_evaluation_deux_var_2d(modele,-5,5,-5,5,niveaux=20)

	return



### Exemple 2 : couche 0 deux neurones, couche 1 : un neurone
def exemple2():
	print("\n--- Exemple 2 : 2+1 neurones ---")
	# Architecture du réseau
	modele = Sequential()
	modele.add(Dense(2, input_dim=2, activation=heaviside, name='Couche_0'))
	modele.add(Dense(1, activation=heaviside, name='Couche_1'))


	# Définition des poids neurone par neurone
	definir_poids(modele,0,0,[-1,3],0)  # definir_poids(modele,couche,rang,coeff,biais)
	definir_poids(modele,0,1,[2,1],0)
	affiche_poids(modele,0)        # affiche poids de la couche 0
	definir_poids(modele,1,0,[1,1],-2)  # definir_poids(modele,couche,rang,coeff,biais)
	affiche_poids(modele,1) 

	affichage_evaluation_deux_var_3d(modele,-5,5,-5,5,num=100)
	affichage_evaluation_deux_var_2d(modele,-5,5,-5,5,num=100,niveaux=20)



	return


# exemple1()
exemple2()
