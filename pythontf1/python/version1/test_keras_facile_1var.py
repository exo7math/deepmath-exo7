#!/usr/bin/python3


################################################
### Exemple d'utilisation du module keras_facile
### Cas d'une variable
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
	modele.add(Dense(1, input_dim=1, activation=heaviside, name='Couche_0'))

	# Définition des poids neurone par neurone
	definir_poids(modele,0,0,-0.5,1)  # definir_poids(modele,couche,rang,coeff,biais)
	affiche_poids(modele,0)        # affiche poids de la couche 0

	# # Exemples d'évaluation
	entree = 0
	# entree = [7.0]
	# entree = np.array([7.0])

	sortie = evaluation(modele,entree)
	print("\n=== Evaluation ===")
	print('Entrée :',entree)
	print('Sortie :',sortie)

	# Affichage graphique de la fonction définie par le réseau sur [a,b] (ici [-5,5])
	affichage_evaluation_une_var(modele,-5,5)
	return



### Exemple 2 : couche 0 deux neurones, couche 1 : un neurone
def exemple2():
	print("\n--- Exemple 2 : 2+1 neurones ---")
	# Architecture du réseau
	modele = Sequential()
	modele.add(Dense(2, input_dim=1, activation=heaviside, name='Couche_0'))
	modele.add(Dense(1, activation='linear', name='Couche_1'))


	# Définition des poids neurone par neurone
	definir_poids(modele,0,0,1,-1)  # definir_poids(modele,couche,rang,coeff,biais)
	definir_poids(modele,0,1,-0.5,1)
	affiche_poids(modele,0)        # affiche poids de la couche 0
	definir_poids(modele,1,0,[2,2],-2)  # definir_poids(modele,couche,rang,coeff,biais)
	affiche_poids(modele,1) 

	# Evaluation
	# entree = [7.0]
	# entree = np.array([7.0])

	entree = 0
	sortie = evaluation(modele,entree)
	print("\n=== Evaluation ===")
	print('Entrée :',entree)
	print('Sortie :',sortie)

	print("Vérification")
	entree = np.array([entree])
	sortie = modele.predict(entree)
	print('Entrée :',entree,'Sortie :',sortie)

	# Affichage graphique
	affichage_evaluation_une_var(modele,-5,5)
	

	return


### Exemple 3 : théorème d'approximation universel
def exemple3():
	# Théorème d'approximation universel
	print("\n--- Exemple 3 : théorème d'approximation universel ---")
	# Fonctions à approcher
	def f(x):
		return np.cos(2*x) + x*np.sin(3*x) + x**0.5

	# Intervalle [a,b] et nombre de divisions
	a = 2
	b = 10
	n = 10

	# Architecture du réseau
	modele = Sequential()

	modele.add(Dense(2*n,input_dim=1,activation=heaviside))
	modele.add(Dense(1,activation='linear'))

	# poids_a_zeros(modele,0)
	# poids_a_zeros(modele,1)

	calcul_approximation(modele,f,a,b,n) # calcule et définis les poids

	# affiche_poids(modele,0)
	# affiche_poids(modele,1)

	affichage_approximation(modele,f,a,b)
	return


# exemple1()
# exemple2()
exemple3()