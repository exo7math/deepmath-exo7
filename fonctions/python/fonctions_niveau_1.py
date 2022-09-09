
# voir fichier 'fonctions_surface_4.sage' pour les calculs exactes


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


n = 150
VX = np.linspace(-2.0, 2.0, n)
VY = np.linspace(-2.0, 2.0, n)
X,Y = np.meshgrid(VX, VY)

def f(x,y):
	z = x**2+y**2
	return z

Z = f(X,Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
# Tweaking display region and labels
# ax.set_xlim(-3.0, 3.0)
# ax.set_ylim(-3.0, 3.0)
# ax.set_zlim(0, 4.0)
ax.set_xlabel('axe x')
ax.set_ylabel('axe y')
ax.set_zlabel('axe z')

ax.view_init(15, -60)

# surface
ax.plot_surface(X, Y, Z, alpha=0.8)

# courbes de niveaux
# faire surface, puis 10 niveaux, puis 20

mes_niveaux = np.linspace(0.05,4,10)
ax.contour(X, Y, Z,mes_niveaux,colors='blue')


# ligne de niveau dans le plan
# fig = plt.figure()
# mes_niveaux = np.linspace(0,3.8,20)
# plt.contour(X, Y, Z,mes_niveaux,colors='red')
# plt.axis('equal') 


plt.tight_layout()
# plt.savefig('fonctions-niveau-1d.png')
plt.show()