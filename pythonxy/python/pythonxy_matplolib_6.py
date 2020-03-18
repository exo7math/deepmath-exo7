
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n = 50
VX = np.linspace(-3.14, 3.14, n)
VY = np.linspace(-3.14, 3.14, n)
X,Y = np.meshgrid(VX, VY)

def f(x,y):  
	z = np.cos(x)*np.sin(y)+2
	return z

Z = f(X,Y)


# Contours et surface
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(40, -60)

mes_niveaux = np.linspace(1,3,10)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none',alpha=0.9)
ax.contour(X, Y, Z, mes_niveaux, czdir='z', offset=1,colors='black', linestyles="solid")

# plt.savefig('pythonxy-niveau-3d-4.png')
plt.show()

