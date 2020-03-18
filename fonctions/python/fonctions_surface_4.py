
# voir fichier 'fonctions_surface_4.sage' pour les calculs exactes

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


n = 50
VX = np.linspace(1, 6.0, n)
VY = np.linspace(0, 5.0, n)
X,Y = np.meshgrid(VX, VY)

def f(x,y):
	z = (x-1)**2+ (y-2)**2 + (x-3)**2 + (y-5)**2 + (x-6)**2 + (y-1)**2
	return z

Z = f(X,Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
# Tweaking display region and labels
# ax.set_xlim(-3.0, 3.0)
# ax.set_ylim(-3.0, 3.0)
# ax.set_zlim(-0.6, 0.6)
ax.set_xlabel('axe x')
ax.set_ylabel('axe y')
ax.set_zlabel('axe z')

ax.view_init(30, -65)

ax.plot_surface(X, Y, Z, alpha = 0.8)

plt.savefig('fonctions-surface-4.png')

plt.tight_layout()
plt.show()