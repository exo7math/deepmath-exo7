
import numpy as np
import matplotlib.pyplot as plt


n = 100
VX = np.linspace(-6.0, 6.0, n)
VY = np.linspace(-6.0, 6.0, n)
X,Y = np.meshgrid(VX, VY)

def f(x,y):
	return np.sin(x)+np.cos(y)

Z = f(X,Y)

# Contour - nb de niveau

nb_niveaux = 10

plt.contour(X, Y, Z,nb_niveaux)

plt.axis('equal') 
# plt.savefig('pythonxy-niveau-2d-1.png')
plt.show()


########### Vue 3D ###########

# from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.view_init(50, -50)
# ax.plot_surface(X, Y, Z)
# plt.savefig('pythonxy-niveau-3d-1.png')
# plt.show()


# Fig 2. Contour - niveaux donnés avec valeurs

# mes_niveaux = np.arange(-1,1,0.2)
# plt.contour(X, Y, Z,mes_niveaux)
# plt.clabel(niveaux, inline=True, fontsize=8)
# plt.savefig('pythonxy-niveau-2.png')


# Fig 3. Contour rempli

# plt.contourf(X, Y, Z,mes_niveaux, cmap='hot') 
# plt.colorbar();
# plt.savefig('pythonxy-niveau-3.png')

# niveaux = plt.contour(X, Y, Z,mes_niveaux, cmap='RdGy') # essayer contour/contourf()
# 


