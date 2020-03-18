import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

N = 100 # Taille de la grille : NxN
# Recherche dans un carré [0,6]x[0,6]
c = 6.0
VX = np.linspace(0, c, N)
VY = np.linspace(0, c, N)
X,Y = np.meshgrid(VX, VY)

def f(x,y):
	# distance au carré
	# z = (x-1)**2+ (y-2)**2 + (x-3)**2 + (y-5)**2 + (x-6)**2 + (y-1)**2
	# distance norme 1
	# z = np.abs(x-1)+ np.abs(y-2) + np.abs(x-3) + np.abs(y-5) + np.abs(x-6) + np.abs(y-1)
	# distance norme 2
	z = np.sqrt((x-1)**2+ (y-2)**2) + np.sqrt((x-3)**2 + (y-5)**2) + np.sqrt((x-6)**2 + (y-1)**2)
	return z

Z = f(X,Y)

Z = f(2.88, 2.99)
print('Z =',Z)

# Minimum sur la grille
print(VX)
print(Z)
print(np.argmin(Z))
imin, jmin = np.unravel_index(np.argmin(Z, axis=None), Z.shape)

print("--- Minimum sur la grille ---")
print("Taille de la grille :",N,"x",N)
print("Nombre de points :",N**2)
print(X[imin,jmin],Y[imin,jmin],Z[imin,jmin])

# Points au hasard
K = 1000
zmin = +np.inf  # infini
for __ in range(K):
	x = c*np.random.random()
	y = c*np.random.random()
	z = f(x,y)
	if z < zmin:
		xmin = x
		ymin = y
		zmin = z

print("--- Minimum au hasard ---")
print("Nombre de points tirés :",K)
print(xmin,ymin,zmin)



# Par les tranches
# voir fichier 'fonctions_surface.6.sage'



