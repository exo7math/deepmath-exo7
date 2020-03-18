
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Partie A - ligne de niveaux
n = 50
VX = np.linspace(-1.0, 1.0, n)
VY = np.linspace(-1.0, 1.0, n)
X,Y = np.meshgrid(VX, VY)

def f(x,y):  # from matlab website
	z = x**2 + y**2
	return z

Z = f(X,Y)


fig = plt.figure()
ax = plt.axes(projection='3d')

# Partie A - Surface
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none',alpha=1)


# Partie B - Opposé du gradients
n = 10
VX = np.linspace(-1.0, 1.0, n)
VY = np.linspace(-1.0, 1.0, n)
U,V = np.meshgrid(VX, VY)

# q = ax.quiver(VX, VY, -U, -V)


# # ax.quiverkey(q, X=0.3, Y=1.1, U=10,
#              label='Quiver key, length = 10', labelpos='E')

ax.view_init(15, -60)
plt.axis('off')
plt.tight_layout()

# plt.savefig('gradient-surface-5c.png')
plt.show()

