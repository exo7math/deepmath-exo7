
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


n = 100
VX = np.linspace(-2.0, 6.0, n)
VY = np.linspace(-2.0, 6.0, n)
X,Y = np.meshgrid(VX, VY)


a1, b1, c1 = -1, 2, 0
a2, b2, c2 = 1, 1, -2
a3, b3, c3 = 0, -1, 3
a4, b4, c4, d4 = 1, 1, 1, -3

# Activations possibles
def sigma(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return np.tanh(x)

def f_heaviside(x):
    if x >= 0:
        return 1
    else:
        return 0
heaviside = np.vectorize(f_heaviside)

def f_relu(x):
    if x >= 0:
        return x
    else:
        return 0
relu = np.vectorize(f_relu)

# choix de la fonction d'activation
# phi = heaviside
# phi = sigma
# phi = tanh
phi = relu  

def F(x,y):
    z = phi(a4*phi(a1*x+b1*y+c1) + b4*phi(a2*x+b2*y+c2) + c4*phi(a3*x+b3*y+c3) + d4)
    return z


Z = F(X,Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('axe x')
ax.set_ylabel('axe y')
ax.set_zlabel('axe z')

ax.view_init(45, -45)
ax.plot_surface(X, Y, Z)
plt.tight_layout()
# plt.savefig('neurones-surface-2-heaviside.png')
# plt.savefig('neurones-surface-2-sigma.png')
# plt.savefig('neurones-surface-2-tanh.png')
# plt.savefig('neurones-surface-2-relu.png')


plt.show()