import numpy as np

# Partie A. Données

# Fonctions à approcher
def f(x,y):
    return x*np.cos(y)

n = 25  # pour le nb de points dans la grille
xmin, xmax, ymin, ymax = -4.0, 4.0, -4.0, 4.0

VX = np.linspace(xmin, xmax, n)
VY = np.linspace(ymin, ymax, n)
X, Y = np.meshgrid(VX, VY)
Z = f(X, Y)

# Affichage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('axe x')
ax.set_ylabel('axe y')
ax.set_zlabel('axe z')

ax.plot_surface(X, Y, Z, color='blue', alpha=0.7)

plt.tight_layout()
ax.view_init(20, -45)
# plt.savefig('tf2-deuxvar-fonction.png')
plt.show()




