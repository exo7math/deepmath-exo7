
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

delta = 0.2
VX = np.arange(-2.5, 2.5, delta)
VY = np.arange(-2.5, 1.7, delta)
X,Y = np.meshgrid(VX, VY)


def f(x,y):
	return y**3 + 2*y**2 - x**2

Z = f(X,Y)

# Graphe de f - figure 3D
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z)

ax.view_init(30, -135)
ax.view_init(5, -145)
# plt.savefig('pythonxy-intro2.png')
plt.show()

# Lignes de niveaux de f - figure 2D
# plt.axis('equal')
# niveaux = np.arange(-2,2,0.5)
# plt.contour(X, Y, Z, niveaux)
# plt.savefig('pythonxy-intro3.png')
# plt.show()

