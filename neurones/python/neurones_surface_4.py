
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


n = 100
VX = np.linspace(-4.0, 3.0, n)
VY = np.linspace(-4.0, 3.0, n)
X,Y = np.meshgrid(VX, VY)


a1, b1, c1 = -1, 2, 0
a2, b2, c2 = 1, 1, -2
a3, b3, c3 = 0, -1, 3
a4, b4, c4, d4 = 1, 1, 1, -3

# Activations possibles
def f_heaviside(x):
    if x >= 0:
        return 1
    else:
        return 0
heaviside = np.vectorize(f_heaviside)

def F(x,y):
    a, b, c = 4, 3, 4
    z = heaviside(a*x+b*y+c)
    return z


Z = F(X,Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('axe x')
ax.set_ylabel('axe y')
ax.set_zlabel('axe z')

ax.view_init(40, -105)
ax.plot_surface(X, Y, Z)
plt.tight_layout()
# plt.savefig('neurones-surface-4.png')

plt.show()