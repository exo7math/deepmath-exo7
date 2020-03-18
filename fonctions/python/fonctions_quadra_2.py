
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n = 50
VX = np.linspace(-2.0, 2.0, n)
VY = np.linspace(-2.0, 2.0, n)
X,Y = np.meshgrid(VX, VY)

def f(x,y):  
	z = x**2
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
# ax.plot_wireframe(X, Y, Z, color='red', rstride=3, cstride=0)  # y = cst

# Courbes de niveaux
mes_niveaux = np.linspace(0.01,3,16)
# ax.contour(X, Y, Z, mes_niveaux, colors='black',linestyles="solid")


plt.tight_layout()
# plt.savefig('fonctions-quadra-2a.png')
plt.show()

# ligne de niveau dans le plan
# fig = plt.figure()
# plt.xlim(-2.0, 2.0)
# plt.ylim(-2.0, 2.0)
# plt.contour(X, Y, Z,mes_niveaux,colors='black')
# plt.axis('equal') 

# plt.tight_layout()
# # plt.savefig('fonctions-quadra-2e.png')
# plt.show()

