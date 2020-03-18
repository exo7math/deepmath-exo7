
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n = 50
VX = np.linspace(-2.0, 2.0, n)
VY = np.linspace(-2.0, 2.0, n)
X,Y = np.meshgrid(VX, VY)

def f(x,y):  
	z = x**2 - y**2
	return z

Z = f(X,Y)


# 
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('axe x')
ax.set_ylabel('axe y')
ax.set_zlabel('axe z')
ax.view_init(15, -120)

# Surface
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, edgecolor='none',alpha=0.9)

#Tranche x = cst
# ax.plot_wireframe(X, Y, Z, color='blue', rstride=0, cstride=2) # x = cst

# Tranche y = cst
# ax.plot_wireframe(X, Y, Z, color='red', rstride=2, cstride=0)  # y = cst

# Courbes de niveaux
# mes_niveaux = np.linspace(-3,-0.1,10)
# ax.contour(X, Y, Z, mes_niveaux, colors='black',linestyles="solid")
# mes_niveaux = np.linspace(0.1,3,10)
# ax.contour(X, Y, Z, mes_niveaux, colors='black',linestyles="solid")
# ax.contour(X, Y, Z, [0], colors='black',linestyles="solid")

plt.tight_layout()
# plt.savefig('fonctions-quadra-3a.png')
plt.show()

# ligne de niveau dans le plan
# fig = plt.figure()
# plt.xlim(-2.0, 2.0)
# plt.ylim(-2.0, 2.0)
# mes_niveaux = np.linspace(0.1,3,10)
# plt.contour(X, Y, Z,mes_niveaux,colors='black')
# mes_niveaux = np.linspace(-3,-0.1,10)
# plt.contour(X, Y, Z,mes_niveaux,colors='black')
# plt.contour(X, Y, Z,[0],colors='black')
# plt.axis('equal') 

# plt.tight_layout()
# # plt.savefig('fonctions-quadra-3e.png')
# plt.show()

