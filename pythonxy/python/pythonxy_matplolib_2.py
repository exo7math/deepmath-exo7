
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# Fig 1. Bosse - grillage

n = 20
VX = np.linspace(-3.0, 3.0, n)
VY = np.linspace(-3.0, 3.0, n)
X,Y = np.meshgrid(VX, VY)

# def f(x,y):
# 	z = 1/(2+x**2+y**2)
# 	return z

# Z = f(X,Y)

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.view_init(40, -70)
# ax.plot_wireframe(X, Y, Z)
# plt.savefig('pythonxy-surface-1.png')

# Fig 2. Quart de plans - hot

# n = 25
# VX = np.linspace(-3.0, 3.0, n)
# VY = np.linspace(-3.0, 3.0, n)
# X,Y = np.meshgrid(VX, VY)


# def f(x,y):
# 	return np.absolute(x)+np.absolute(y)

# Z = f(X,Y)

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.view_init(40, -70)
# ma_surface = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot', edgecolor='none')
# fig.colorbar(ma_surface)
# plt.savefig('pythonxy-surface-2.png')



# Fig 3. sin(r) # couleur

n = 50
VX = np.linspace(-6.0, 6.0, n)
VY = np.linspace(-6.0, 6.0, n)
X,Y = np.meshgrid(VX, VY)

def f(x,y):
	r = np.sqrt(x**2+y**2)
	z = np.sin(r)
	return z

Z = f(X,Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(40, -120)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
# plt.savefig('pythonxy-surface-3.png')





# ax.view_init(45, -70)
# ax.plot_wireframe(X, Y, Z)
# ax.plot_surface(X, Y, Z)
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
#                 cmap='hot', edgecolor='none')
# cmap='binary'
# cmap='viridis'

# ax.contour(X, Y, Z, colors='black', linewidths=3)
# ma_surface = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot', edgecolor='none',alpha=0.5)

# ax.contour(X, Y, Z, zdir='z', offset=-2, cmap='hot')

# fig.colorbar(ma_surface);

plt.show()