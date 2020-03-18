
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------
# 3. Axes

def f(x):
	return np.exp(-x) * np.cos(2*np.pi*x)

a,b = 0,5
X = np.linspace(a,b,num=100)
Y = f(X)

plt.title('Amorti') # titre
plt.axis('equal')   # repère orthonormé
plt.grid()          # grille 
plt.xlim(a,b)       # bornes de l'axe des x
plt.plot(X,Y)
plt.savefig('pythonx-amorti.png')
plt.show()


# X1 = np.random.random(10)
# Y1 = np.random.random(10)
# X2 = np.random.random(10)
# Y2 = np.random.random(10)
# plt.scatter(X1,Y1, marker='o',color='blue')
# plt.scatter(X2,Y2, marker='s',color='red')
# plt.xlim(0,1)
# plt.ylim(0,1)
# plt.xticks([])    # sans les valeurs x
# plt.yticks([])
# plt.show()



