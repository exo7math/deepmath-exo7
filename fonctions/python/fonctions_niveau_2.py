
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n = 50
VX = np.linspace(-2.0, 2.0, n)
VY = np.linspace(-2.0, 2.0, n)
X,Y = np.meshgrid(VX, VY)

def f(x,y):  
	r = np.sqrt(x**2+y**2)
	z = np.sin(r**2-x)/r+1
	return z

Z = f(X,Y)


# Contours et surface
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('axe x')
ax.set_ylabel('axe y')
ax.set_zlabel('axe z')
ax.view_init(25, -25)

mes_niveaux = np.linspace(0,3,20)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, edgecolor='none',alpha=0.9)
ax.contour(X, Y, Z, mes_niveaux, offset=0,colors='black', linestyles="solid")


plt.tight_layout()
# plt.savefig('fonctions-niveau-2a.png')
plt.show()

# ligne de niveau dans le plan
fig = plt.figure()
plt.xlim(-2.0, 2.0)
plt.ylim(-2.0, 2.0)
plt.contour(X, Y, Z,mes_niveaux,colors='black')
plt.axis('equal') 

# plt.tight_layout()
# plt.savefig('fonctions-niveau-2b.png')
# plt.show()

