
import numpy as np
import matplotlib.pyplot as plt


n = 50
VX = np.linspace(-3.0, 3.0, n)
VY = np.linspace(-3.0, 3.0, n)
X,Y = np.meshgrid(VX, VY)

def f(x,y):  # from matlab website
	z = np.sin(2*x+y**2)
	return z

Z = f(X,Y)


# Contour - niveaux donnés avec valeurs

mes_niveaux = np.linspace(-1,1,11)
plt.contourf(X, Y, Z,mes_niveaux, cmap='hot') 
plt.colorbar();
plt.contour(X, Y, Z,mes_niveaux)

plt.axis('equal') 
# plt.savefig('pythonxy-niveau-2d-3.png')
plt.show()


########### Vue 3D ###########

# from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.view_init(50, -50)
# ax.plot_surface(X, Y, Z,cmap=plt.cm.CMRmap)
# # ax.plot_surface(X, Y, Z,cmap=plt.cm.Spectral)
# # plt.savefig('pythonxy-niveau-3d-3.png')
# plt.show()



