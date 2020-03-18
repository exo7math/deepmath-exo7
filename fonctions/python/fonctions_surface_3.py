
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


n = 100
VX = np.linspace(-3.0, 5.0, n)
VY = np.linspace(-3.0, 4.0, n)
X,Y = np.meshgrid(VX, VY)

def f(x,y):
	r1 = x**2 + y**2
	r2 = (x-2)**2 + (y-1)**2
	z = -x * np.exp(-r1) - 1/3*y**2 * np.exp(-r2) 
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

ax.view_init(15, -75)

ax.plot_surface(X, Y, Z)


plt.tight_layout()
# plt.savefig('fonctions-surface-3.png')
plt.show()