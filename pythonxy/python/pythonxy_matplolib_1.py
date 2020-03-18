
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


n = 5
VX = np.linspace(-2.0, 2.0, n)
VY = np.linspace(-2.0, 2.0, n)
X,Y = np.meshgrid(VX, VY)

def f(x,y):
	return x**2-y**2

Z = f(X,Y)

fig = plt.figure()
ax = plt.axes(projection='3d')

# Fig 1. n = 5
ax.view_init(40, -30)
ax.plot_surface(X, Y, Z)
# plt.savefig('pythonxy-selle-1.png')

# Fig 2. n = 10
# ax.view_init(35, -50)
# ax.plot_surface(X, Y, Z)
# plt.savefig('pythonxy-selle-2.png')

# Fig 3. n = 20
# ax.view_init(30, -70)
# ax.plot_surface(X, Y, Z)
# plt.savefig('pythonxy-selle-3.png')


plt.show()