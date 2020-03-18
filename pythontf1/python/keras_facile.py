##########################################
#####      Module : keras_facile     #####
#####                                #####
##### Auteur : Arnaud Bodin          #####
##### Date : Janvier 2020            #####
##########################################

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Fonction d'activation Heaviside
from tensorflow.keras import backend as K
def heaviside(x):
	""" Définis la fonction de Heaviside qui n'est pas défini
	par défaut dans keras. A utiliser comme fonction 
	d'activiation lors de la définition d'une couche par exemple
	    modele.add(Dense(4,activation=heaviside))
	Attention il n'y a pas de guillemet ici.
	Astuce de la formule : H(x) = 1/2 (1+|x|) """

	# return (1+x/K.abs(x))/2
	# return (1+K.sign(x))/2

	z = K.zeros_like(x)
	return 1-K.maximum(z,K.sign(-x))


# Affichage des poids
def affiche_poids(modele,couche):
	""" Affiches les informations d'un couche donnée :
	  - nb de neurones, nb d'entrée, nb de poids par neurones.
	  Puis pour chaque neurone, affiche du poids : coeff + biais. """


	coeff, biais = modele.layers[couche].get_weights()
	nb_entrees = len(coeff)
	nb_neurones = len(coeff[0])
	print("\n\n===== Couche numéro",couche,"=====")
	print("Nombre de neurones :",nb_neurones)
	print("Nombre d'entrées par neurones :",nb_entrees)
	print("Nombre de poids par neurones :",nb_entrees+1,"\n")	
	for i in range(nb_neurones):
		print("    --- Neurone numéro",i,"---")
		print('    Coefficients',np.transpose(coeff)[i])
		print('    Biais',biais[i])
	return


# Mettre tous les poids à zéros (purement esthétique)
def poids_a_zeros(modele,couche):
	""" Met tous les poids à zéros d'une couche.
	C'est par soucis esthétique car lors de la définition d'une couche, 
	les poids initiaux sont des nombres au hasard. """

	coeff, biais = modele.layers[couche].get_weights()
	newcoeff = np.zeros(np.shape(coeff))
	newbiais = np.zeros(np.shape(biais))
	newpoids = [newcoeff,newbiais]
	modele.layers[couche].set_weights(newpoids)
	return

# Définir les poids d'un neurone à la main
def definir_poids(modele,couche,rang,ncoeff,nbiais):
	""" Définis les poids d'un neurone.
	Le neurone est identifié par la couche et le rang dans cette couche.
	ncoeff : les coefficients à définir pour le neurone, 
	cela peut être un réel (si un seul coeff), une liste de réels, 
	ou un vecteur numpy.
	nbiais : un réel 
	Exemples :
	definir_poids(modele,0,2,7,3)  # si une entrée
	definir_poids(modele,0,2,[7,-2],3)  # si deux entrées (ou plus)
	definir_poids(modele,0,2,vecteur,3)  # où vecteur est vecteur numpy
	"""

	# Récupérer les poids actuels
	coeff, biais = modele.layers[couche].get_weights()

	# Changer le biais d'un neurone
	biais[rang] = nbiais

	#Chager les coeff d'un neurone
	if isinstance(ncoeff, (int, float, complex)):  # cas d'une seule valeur 
		ncoeff = [ncoeff]  #  x devient [x]
		
	if isinstance(ncoeff, list):
		ncoeff = np.array(ncoeff)  # Transforme liste en array si besoin

	nb_entrees = len(coeff)
	for i in range(nb_entrees):
		coeff[i,rang] = ncoeff[i]
	newpoids = [coeff,biais]
	modele.layers[couche].set_weights(newpoids)
	
	return 


# Evaluation d'un réseau (variante de predict())
def evaluation(modele,*entree):
	""" Renvoie le résultat calculer par le réseau de neurones en fonction de l'entrée.
	'entree' peut être un réel (si un seul coeff), une liste de réels, 
	ou un vecteur numpy.
	Exemples :	
	evaluation(modele,7)  # une entrée
	evaluation(modele,[10,5,7]) # deux entrée ou plus (ici trois)
	evaluation(modele,vecteur)  # avec vecteur numpy   """

	# Cas x ou x,y ou x,y,z
	if isinstance(entree[0], (int, float, complex)):  
		entree = list(entree)
	else:
		entree = entree[0]	

	# predict doit recevoir tableau numpy type [[x,y,z]]
	entree =  np.array([entree])
	sortie = modele.predict(entree)
	if len(sortie)==1:
		sortie = sortie[0]
	return sortie[0]


# Graphique une variable
def affichage_evaluation_une_var(modele,a,b,rang=0,num=100):
	""" Affichage graphique dans le cas où une seul entrée au réseau.
	La fonction associée au neurone dont le rang est donné (par défaut le premier neurone)
	est tracée sur l'intervalle [a,b], divisé en n """
	# Affichage graphique
	liste_x = np.linspace(a, b, num=num)
	entree =  np.array([[x] for x in liste_x])
	sortie = modele.predict(entree)
	liste_y = np.array([y[rang] for y in sortie])
	plt.plot(liste_x,liste_y)
	plt.tight_layout()
	# plt.savefig('pythontf-1var.png')
	plt.show()


# Graphique du théorème d'approximation universel
def affichage_approximation(modele,f,a,b,rang=0,num=100):
	""" Comme 'affichage_evaluation_une_var' 
	mais affiche en plus le graphe de f """
	liste_x = np.linspace(a, b, num=num)
	entree =  np.array([[x] for x in liste_x])
	sortie = modele.predict(entree)
	liste_y = np.array([y[rang] for y in sortie])
	plt.plot(liste_x,f(liste_x),color='red')
	plt.plot(liste_x,liste_y,color='blue',linewidth=2)
	plt.tight_layout()
	# plt.savefig('pythontf-approx.png')
	plt.show()


# Définis un réseau qui approxime une fonction 
def calcul_approximation(modele,f,a,b,n): 
# calcule et définis les poids

	# Couche 0 : 2*n neurones
	h = (b-a)/n
	x = a
	xx = a+h
	liste_y = []
	for i in range(n):	
		definir_poids(modele,0,2*i,1/x,-1)
		definir_poids(modele,0,2*i+1,-1/xx,1)

		y = f(x)
		liste_y = liste_y + [y,y]

		x = x + h
		xx = xx + h

	# Couche 1 : un seul neurone
	definir_poids(modele,1,0,liste_y,-sum(liste_y)/2)
	return


# Graphique 3d deux variables
def affichage_evaluation_deux_var_3d(modele,xmin,xmax,ymin,ymax,rang=0,num=20):
	""" Affichage graphique dans le cas où une seul entrée au réseau.
	La fonction associée au neurone dont le rang est donnée (par défaut le premier neurone)
	est tracée sur la zone [xmin,xmax]x[ymin,ymax], divisé en n """
	
	VX = np.linspace(xmin, xmax, num)
	VY = np.linspace(ymin, ymax, num)

	X,Y = np.meshgrid(VX, VY)

	entree = np.c_[X.ravel(), Y.ravel()]

	sortie = modele.predict(entree)
	Z = sortie.reshape(X.shape)

	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.set_xlabel('axe x')
	ax.set_ylabel('axe y')
	ax.set_zlabel('axe z')

	ax.plot_surface(X, Y, Z)

	plt.tight_layout()
	# ax.view_init(50, -110)
	# plt.savefig('pythontf-2var-3d.png')
	plt.show()

	return

# Graphique 2d deux variables
def affichage_evaluation_deux_var_2d(modele,xmin,xmax,ymin,ymax,rang=0,num=30,niveaux=10):
	""" Affichage graphique dans le cas où une seul entrée au réseau.
	La fonction associée au neurone dont le rang est donnée (par défaut le premier neurone)
	est tracée sur la zone [xmin,xmax]x[ymin,ymax], divisé en n """

	VX = np.linspace(xmin, xmax, num)
	VY = np.linspace(ymin, ymax, num)	

	X,Y = np.meshgrid(VX, VY)
	entree = np.c_[X.ravel(), Y.ravel()]

	sortie = modele.predict(entree)
	Z = sortie.reshape(X.shape)

	fig = plt.figure()

	plt.contourf(X, Y, Z,niveaux, cmap='hot')
	plt.colorbar();
	plt.axis('equal') 
	plt.tight_layout()
	# plt.savefig('pythontf-2var-2d.png')
	plt.show()

	return


# Graphique 3d trois variables
def affichage_evaluation_trois_var(modele,xmin,xmax,ymin,ymax,zmin,zmax,rang=0,num=10):
	""" Affichage graphique dans le cas où une seul entrée au réseau.
	La fonction associée au neurone dont le rang est donnée (par défaut le premier neurone)
	est tracée sur la zone [xmin,xmax]x[ymin,ymax], divisé en n """
	
	VX = np.linspace(xmin, xmax, num)
	VY = np.linspace(ymin, ymax, num)
	VZ = np.linspace(zmin, zmax, num)

	X,Y,Z = np.meshgrid(VX, VY, VZ)
	entree = np.c_[X.ravel(), Y.ravel(), Z.ravel()]

	sortie = modele.predict(entree)
	S = sortie.reshape(X.shape)

	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.set_xlabel('axe x')
	ax.set_ylabel('axe y')
	ax.set_zlabel('axe z')

	points = ax.scatter(X.ravel(),Y.ravel(),Z.ravel(), c=S.ravel(), cmap='hot')
	fig.colorbar(points)
	plt.tight_layout()
	# ax.view_init(20, 55)
	# plt.savefig('pythontf-3var.png')	
	plt.show()

	return


# Références 'mpariente' https://stackoverflow.com/questions/51140950/
def get_weights_grad(model, inputs, outputs):
    """ Gets gradient of model for given inputs and outputs for all weights """
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    return output_grad

