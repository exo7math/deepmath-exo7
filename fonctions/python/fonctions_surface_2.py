
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


n = 50
VX = np.linspace(-5.0, 5.0, n)
VY = np.linspace(-5.0, 5.0, n)
X,Y = np.meshgrid(VX, VY)

def f(x,y):
	return x*np.sin(y)

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

ax.view_init(40, -55)    # a et c et e
# ax.view_init(40, -75)    # b
# ax.view_init(40, -10)     # d


ax.plot_wireframe(X, Y, Z, color='blue', rstride=0, cstride=2) # x = cst
ax.plot_wireframe(X, Y, Z, color='red', rstride=2, cstride=0)  # y = cst

# ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2) # NON : mettre les deux
# ax.plot_surface(X, Y, Z)


plt.savefig('fonctions-surface-2e.png')

plt.tight_layout()
plt.show()