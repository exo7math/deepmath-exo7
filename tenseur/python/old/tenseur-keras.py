
# Tenseur avec tensorflow

import tensorflow as tf
import numpy as np

from tensorflow.keras import backend as K

# Partie A - Une case vide

# Fonction x -> x + 2

x = K.placeholder(shape=(1,))
y = x + 2 
f = K.function([x],[y])
print(f([1]))

X = np.array([[1],[2],[3]])
print(f(X))


# Fonction produit scalaire
U = K.placeholder(shape=(3,))
V = K.placeholder(shape=(3,))
PS = sum(U[i]*V[i] for i in range(3))
f = K.function([U,V], [PS])

valU = np.array([1,2,3])
valV = np.array([4,5,6])
valPS = f([valU, valV])
print("Produit scalaire :", valPS)


# Produit matrice x vecteur
A = K.placeholder(shape=(3,3))
X = K.placeholder(shape=(3,1))
Y = K.dot(A,X)
f = K.function([A, X], [Y])

Aval = np.arange(1,10).reshape((3,3))
Xval = np.arange(1,4).reshape((3,1))

Yval = f([Aval, Xval])

print("Produit matrice x vecteur :\n", Yval)


# Partie B - Gradient

# Une variable (en un seul points)
x = K.constant([3])
y = x**2
grad = K.gradients(y, x)
print("Dérivée :", K.eval(grad[0]))


# Une variable (en plusieurs points)
x = K.constant([1,2,3,4])
y = x**2
grad = K.gradients(y, x)
print("Dérivée :", K.eval(grad[0]))


# Fonction personnalisée
def f(x):
	return 2*K.log(x) + K.exp(-x)

X = K.arange(1,4,0.5)
Y = f(X)
grad = K.gradients(Y, X)

print("Point x :", K.eval(X))
print("Valeur y=f(x) :", K.eval(Y))
print("Dérivée f'(x) :", K.eval(grad[0]))


# Affichage
import matplotlib.pyplot as plt

def affichage_derivee(f,a,b,epsilon=0.01):
	X = K.arange(a,b,epsilon)
	Y = f(X)
	dY = K.gradients(Y, X)[0]
	Xnp, Ynp, dYnp = K.eval(X), K.eval(Y), K.eval(dY)
	plt.plot(Xnp, Ynp, color='blue', label="f(x)")
	plt.plot(Xnp, dYnp, color='red', label="f'(x)")
	plt.legend()
	plt.tight_layout()
	# plt.savefig('tenseur-derivee.png')
	plt.show()
	return


# def f(x):
# 	return K.log(x)

# affichage_derivee(f,1,4)


# Partie C - Gradient deux variables

# Deux variables - Gradient en un point
x = K.constant([2])
y = K.constant([3])
z = x * (y**2) 
grad = K.gradients(z, [x,y])
dxZ = K.eval(grad[0])
dyZ = K.eval(grad[1])
print("Gradient :", dxZ, dyZ)

# Deux variables - Gradient sur une grille

# Grille numpy
n = 100
VX = np.linspace(-2.0, 2.0, n)
VY = np.linspace(-2.0, 2.0, n)
Xnp,Ynp = np.meshgrid(VX, VY)

# Grille keras
X, Y = K.constant(Xnp), K.constant(Ynp)

# Fonction
def f(x,y):
	return x**2-y**2

Z = f(X,Y) 
grad = K.gradients(Z, [X,Y])
dxZ = K.eval(grad[0])
dyZ = K.eval(grad[1])

# print("Gradient :", dxZ, dyZ)

# Lignes de niveau
Znp = K.eval(Z)
plt.contour(Xnp, Ynp, Znp)
plt.axis('equal') 

# Gradient
step = 10  # pour éviter d'afficher toute les flèches
plt.quiver(Xnp[::step,::step],Ynp[::step,::step],dxZ[::step,::step],dyZ[::step,::step])

# plt.savefig('tenseur-gradient.png')
plt.show()