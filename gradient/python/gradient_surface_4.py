
import numpy as np
import matplotlib.pyplot as plt

# Partie A - ligne de niveaux
n = 100
VX = np.linspace(-1.0, 1.0, n)
VY = np.linspace(-1.0, 1.0, n)
X,Y = np.meshgrid(VX, VY)

def f(x,y):  
	z = x**2 - y**2
	return z

Z = f(X,Y)


# Contour - niveaux donnés avec valeurs

mes_niveaux = np.linspace(-1,1,11)
mes_niveaux = np.append(mes_niveaux,0.1)
mes_niveaux = np.append(mes_niveaux,-0.1)
mes_niveaux = np.sort(mes_niveaux)
plt.contour(X, Y, Z, mes_niveaux)


# Partie B - opposé du gradients
n = 10
VX = np.linspace(-1.0, 1.0, n)
VY = np.linspace(-1.0, 1.0, n)
U,V = np.meshgrid(VX, VY)

q = plt.quiver(VX, VY, -U, V)


plt.quiverkey(q, X=0.3, Y=1.1, U=10,
             label='Quiver key, length = 10', labelpos='E')

plt.axis('equal') 
plt.axis('off')
plt.tight_layout()

# plt.savefig('gradient-surface-5c.png')
plt.show()

