
import numpy as np
import matplotlib.pyplot as plt


n = 200
VX = np.linspace(-0.2, 0.2, n)
VY = np.linspace(-0.3, 0.1, n)
X,Y = np.meshgrid(VX, VY)

def f(x,y):  # from matlab website
	z = x**2 - y**3 + x*y
	return z

Z = f(X,Y)


# Contour - niveaux donnés avec valeurs

mes_niveaux = np.linspace(-0.1,0.1,101)
trace = plt.contour(X, Y, Z, mes_niveaux)
# plt.clabel(trace, inline=True, fontsize=8)


plt.axis('equal') 
plt.tight_layout()

# plt.savefig('gradient-surface-6.png')
plt.show()

