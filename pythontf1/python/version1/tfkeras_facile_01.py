#!/usr/bin/python3

# Comme 'tf_keras_01.py' 
# mais avec module keras_facile

from keras_facile import *

modele = Sequential()
modele.add(Dense(2, input_dim=1, activation='relu'))
modele.add(Dense(1, activation='relu'))

modele.summary()

# Poids de la couche 0
definir_poids(modele,0,0,1,-1)     # definir_poids(modele,couche,rang,coeff,biais)
definir_poids(modele,0,1,-0.5,1)
affiche_poids(modele,0)            # affiche poids de la couche 0

# Poids de la couche 1
definir_poids(modele,1,0,[1,1],0) 
affiche_poids(modele,1) 

# Evaluation
entree = 3
sortie = evaluation(modele,entree)
print('Entrée :',entree,'Sortie :',sortie)

# Affichage graphique
affichage_evaluation_une_var(modele,-2,3)

