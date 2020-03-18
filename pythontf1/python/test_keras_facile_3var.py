#!/usr/bin/python3


################################################
### Exemple d'utilisation du module keras_facile
### Cas de trois variables
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
	modele.add(Dense(1, input_dim=3, activation='relu', name='Couche_0'))

	# Définition des poids neurone par neurone
	definir_poids(modele,0,0,[1,-1,1],-1)  # definir_poids(modele,couche,rang,coeff,biais)
	affiche_poids(modele,0)        # affiche poids de la couche 0

	# # Exemples d'évaluation
	entree = [2.0,3.0,4.0]
	# entree = np.array([2.0, 3.0, 4.0])

	sortie = evaluation(modele,entree)
	print("\n=== Evaluation ===")
	print('Entrée :',entree)
	print('Sortie :',sortie)

	# Affichage graphique de la fonction 
	affichage_evaluation_trois_var(modele,-5,5,-5,5,-5,5,num=10)

	return



### Exemple 2 : un cube, couche 0 : 6 neurones, couche 1 : un neurone
def exemple2():
	print("\n--- Exemple 2 : 6+1 neurones ---")
	# Architecture du réseau
	modele = Sequential()
	modele.add(Dense(6, input_dim=3, activation='sigmoid', name='Couche_0'))
	modele.add(Dense(1, activation='sigmoid', name='Couche_1'))


	# Définition des poids neurone par neurone
	definir_poids(modele,0,0,[1,0,0],1)
	definir_poids(modele,0,1,[-1,0,0],1)
	definir_poids(modele,0,2,[0,1,0],1)
	definir_poids(modele,0,3,[0,-1,0],1)
	definir_poids(modele,0,4,[0,0,1],1)
	definir_poids(modele,0,5,[0,0,-1],1)
	affiche_poids(modele,0)        
	definir_poids(modele,1,0,[1,1,1,1,1,1],-6) 
	affiche_poids(modele,1) 

	affichage_evaluation_trois_var(modele,-3,3,-3,3,-3,3,num=12)


	return


# exemple1()
exemple2()
