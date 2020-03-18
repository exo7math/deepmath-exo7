
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Partie A - ligne de niveaux
n = 50
VX = np.linspace(-0.2, 0.15, n)
VY = np.linspace(-0.3, 0.1, n)
X,Y = np.meshgrid(VX, VY)

def f(x,y):  # from matlab website
	z = x**2 - y**3 + x*y
	return z

Z = f(X,Y)


fig = plt.figure()
ax = plt.axes(projection='3d')

# Partie A - Surface
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none',alpha=1)


ax.set_xlabel('axe x')
ax.set_ylabel('axe y')
ax.set_zlabel('axe z')
ax.view_init(15, -60)

plt.tight_layout()

# plt.savefig('gradient-surface-6.png')
plt.show()

