#!/usr/bin/python3

# Comme 'tf_keras_02.py' 
# mais avec module keras_facile

from keras_facile import *

modele = Sequential()
modele.add(Dense(3, input_dim=2, activation='sigmoid'))
modele.add(Dense(1, activation='sigmoid'))

# Poids de la couche 0
definir_poids(modele,0,0,[1.0,2.0],-1.0)     # definir_poids(modele,couche,rang,coeff,biais)
definir_poids(modele,0,1,[3.0,-4.0],0.0)
definir_poids(modele,0,2,[-5.0,-6.0],1.0)
affiche_poids(modele,0)                      # affiche poids de la couche 0

# Poids de la couche 1
definir_poids(modele,1,0,[1.0,1.0,1.0],-3.0) 
affiche_poids(modele,1) 

# Evaluation
entree = [7.0,-5.0]
sortie = evaluation(modele,entree)
print('Entrée :',entree,'Sortie :',sortie)

# Affichage graphique
affichage_evaluation_deux_var_3d(modele,-5,5,-5,5)

