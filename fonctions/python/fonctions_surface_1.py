
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


n = 50
VX = np.linspace(-3.0, 3.0, n)
VY = np.linspace(-3.0, 3.0, n)
X,Y = np.meshgrid(VX, VY)

def f(x,y):
	return x*np.exp(-x**2-y**2)

Z = f(X,Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
# Tweaking display region and labels
# ax.set_xlim(-3.0, 3.0)
# ax.set_ylim(-3.0, 3.0)
ax.set_zlim(-0.6, 0.6)
ax.set_xlabel('axe x')
ax.set_ylabel('axe y')
ax.set_zlabel('axe z')

ax.view_init(15, -90)
# ax.view_init(25, -70)
# ax.view_init(35, -50)
# ax.view_init(45, -30)

ax.plot_surface(X, Y, Z)
plt.tight_layout()
plt.savefig('fonctions-surface-1a.png')

# trouver une séquence de vues

plt.show()