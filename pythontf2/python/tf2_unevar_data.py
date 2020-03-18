import numpy as np

# Données

# Fonctions à approcher
def f(x):
    return np.cos(2*x) + x*np.sin(3*x) + x**0.5 - 2

a, b = 0, 5                 # intervalle [a,b]
N = 100                     # taille des données
X = np.linspace(a, b, N)    # abscisses
Y = f(X)                    # ordonnées
X_train = X.reshape(-1,1)
Y_train = Y.reshape(-1,1)


# Affichage de la fonction
import matplotlib.pyplot as plt

plt.plot(X_train, Y_train, color='blue')
# plt.savefig('unevar-fonction.png')
plt.show()

