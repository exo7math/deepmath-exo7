
import numpy as np
import matplotlib.pyplot as plt


n = 100
VX = np.linspace(-3.0, 3.0, n)
VY = np.linspace(-3.0, 3.0, n)
X,Y = np.meshgrid(VX, VY)

def f(x,y):  # from matlab website
	z = 3 * (x-1)**2 * np.exp(-(y+1)**2-x**2) - 1/3 * (np.exp(-(x+1)**2-x**2)) + (10*x**3-2*x+10*y**5)*np.exp(-x**2-y**2)
	return z

Z = f(X,Y)


# Contour - niveaux donnés avec valeurs

mes_niveaux = np.arange(-5,5,1)
trace = plt.contour(X, Y, Z, mes_niveaux)
plt.clabel(trace, inline=True, fontsize=8)
plt.axis('equal') 
# plt.savefig('pythonxy-niveau-2d-2.png')
plt.show()


########### Vue 3D ###########

# from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.view_init(50, -50)
# ax.plot_surface(X, Y, Z, alpha=1)
# plt.savefig('pythonxy-niveau-3d-2.png')
# plt.show()



ax.plot_surface(X, Y, Z,cmap=plt.cm.CMRmap)

ax.plot_surface(X, Y, Z,cmap=plt.cm.Spectral)

# Fig 3. Contour rempli

# plt.contourf(X, Y, Z,mes_niveaux, cmap='hot') 
# plt.colorbar();
# plt.savefig('pythonxy-niveau-3.png')

# niveaux = plt.contour(X, Y, Z,mes_niveaux, cmap='RdGy') # essayer contour/contourf()
# 


