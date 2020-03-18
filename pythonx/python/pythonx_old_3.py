
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------
# Courbes de Lissajou 

# BOF - trop techniwue - à garder pour les TD (sans faire de grille)

numpoints = 100

T = np.linspace(0,2*np.pi,numpoints)
X = []
Y = []


for a in range(3,7):
	for b in range(2,5):
		X = X + [np.sin(a*T+0*np.pi/2)]
		Y = Y + [np.sin(b*T)]
		ax = plt.subplot(4,3,(a-3)*3+(b-2)+1)
		ax.set(aspect='equal')
		ax.set_title("a = {}, b = {}".format(a,b))
		plt.plot(X[a-3],Y[b-2])

plt.savefig('lissajou.png')
plt.savefig('lissajou.pdf')
plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.8,wspace=0.5)
plt.show()
