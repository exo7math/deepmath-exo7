
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


n = 100
VX = np.linspace(-3.0, 6.0, n)
VY = np.linspace(-3.0, 6.0, n)
X,Y = np.meshgrid(VX, VY)




# Activations possibles
def sigma(x):
    return 1/(1+np.exp(-x))

def F(x,y):
    a, b, c = 1, 2, -3
    z = sigma(a*x+b*y+c)
    return z


Z = F(X,Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('axe x')
ax.set_ylabel('axe y')
ax.set_zlabel('axe z')

ax.view_init(35, -50)
ax.plot_surface(X, Y, Z)
plt.tight_layout()
plt.savefig('neurones-surface-3.png')


plt.show()