
import numpy as np
import matplotlib.pyplot as plt


n = 100
VX = np.linspace(-1.0, 1.0, n)
VY = np.linspace(-1.0, 1.0, n)
X,Y = np.meshgrid(VX, VY)

def f(x,y):  # from matlab website
	z = x**2 - y**2
	return z

Z = f(X,Y)


# Contour - niveaux donnés avec valeurs

mes_niveaux = np.linspace(-1,1,11)
mes_niveaux = np.append(mes_niveaux,0.1)
mes_niveaux = np.append(mes_niveaux,-0.1)
mes_niveaux = np.sort(mes_niveaux)
trace = plt.contour(X, Y, Z, mes_niveaux)
plt.clabel(trace, inline=True, fontsize=8)


plt.axis('equal') 
plt.tight_layout()

# plt.savefig('gradient-surface-4.png')
plt.show()

