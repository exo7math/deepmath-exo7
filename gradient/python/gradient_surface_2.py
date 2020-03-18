
# voir fichier 'fonctions_surface_4.sage' pour les calculs exactes

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n = 50
VX = np.linspace(-1, 1, n)
VY = np.linspace(-1, 1, n)
X,Y = np.meshgrid(VX, VY)

def f(x,y):
    # z = x**2 + y**2
    # z = -x**2 - y**2
    z = x**2 - y**2
    return z

Z = f(X,Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
plt.axis('off')
# Tweaking display region and labels
# ax.set_xlim(-3.0, 3.0)
# ax.set_ylim(-3.0, 3.0)
# ax.set_zlim(-0.6, 0.6)
# ax.set_xlabel('axe x')
# ax.set_ylabel('axe y')
# ax.set_zlabel('axe z')

ax.view_init(15, -60)
# ax.view_init(15, -75)
# ax.view_init(15, -105)

# ax.plot_surface(X, Y, Z, alpha = 0.8)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none',alpha=1)
# mes_niveaux = np.linspace(-0.1,1,7)
# ax.contour(X, Y, Z, mes_niveaux, czdir='z', offset=-0.1,colors='black', linestyles="solid")


plt.tight_layout()

# plt.savefig('gradient-surface-3a.png')

plt.show()