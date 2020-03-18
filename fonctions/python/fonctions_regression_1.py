
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


N = 100
VX = np.linspace(0.1, 1.5, N)
VY = np.linspace(0.1, 5.0, N)
X,Y = np.meshgrid(VX, VY)



def f(a,b):
	# objectifs du genre y = 0.5x+1.5
	points = [(2,3),(3,5),(4,4),(5,6),(6,6)]

	z = 0
	for P in points:
		z = z + (a*P[0]-P[1]+b)**2
	return z/(1+a**2)

Z = f(X,Y)



def affiche_surface():
	fig = plt.figure()
	ax = plt.axes(projection='3d')
	# Tweaking display region and labels
	# ax.set_xlim(-3.0, 3.0)
	# ax.set_ylim(-3.0, 3.0)
	# ax.set_zlim(-0.6, 0.6)
	ax.set_xlabel('axe x')
	ax.set_ylabel('axe y')
	ax.set_zlabel('axe z')

	ax.view_init(40, -30)

	ax.plot_surface(X, Y, Z)

	plt.tight_layout()
	# plt.savefig('fonctions-surface-3a.png')

	# trouver une séquence de vues

	plt.show()
	return


def affiche_niveaux():
	fig = plt.figure()

	mes_niveaux = np.linspace(0,5,20)
	plt.contour(X, Y, Z, mes_niveaux, colors='black')

	plt.scatter(0.79,1.61,color='blue')

	plt.xlabel('axe a')
	plt.ylabel('axe b')
	plt.tight_layout()
	# plt.savefig('fonctions-regression.png')

	plt.show()
	return	


def cherche_minimum():
	# Minimum sur la grille

	imin, jmin = np.unravel_index(np.argmin(Z, axis=None), Z.shape)

	print("--- Minimum sur la grille ---")
	print("Taille de la grille :",N,"x",N)
	print("Nombre de points :",N**2)
	print(X[imin,jmin],Y[imin,jmin],Z[imin,jmin])
	return

# affiche_surface()

affiche_niveaux()
 
cherche_minimum()
# a = 0.79
# b = 1.61
# f = 1.219

