import numpy as np
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Partie A. Données

# Fonctions à approcher
def f(x,y):
    return x*np.cos(y)

n = 25  # pour le nb de points dans la grille
xmin, xmax, ymin, ymax = -4.0, 4.0, -4.0, 4.0

VX = np.linspace(xmin, xmax, n)
VY = np.linspace(ymin, ymax, n)
X, Y = np.meshgrid(VX, VY)
Z = f(X, Y)

entree = np.append(X.reshape(-1,1), Y.reshape(-1,1), axis=1)
sortie = Z.reshape(-1, 1)

# Partie B. Réseau 

modele = Sequential()

p = 10
modele.add(Dense(p, input_dim=2, activation='relu'))
modele.add(Dense(p, activation='relu'))
modele.add(Dense(p, activation='relu'))
modele.add(Dense(1, activation='linear'))


# Méthode de gradient : descente de gradient stochastique avec taux d'apprentissage donné
# Fonction d'erreur : erreur moyenne quadratique
mysgd = optimizers.SGD(lr=0.01)
modele.compile(loss='mean_squared_error', optimizer=mysgd)

# Affiche un résumé
print(modele.summary())

# Partie C. Apprentissage

# modele.fit(entree, sortie, epochs=1000, batch_size=len(entree), verbose = 1)

# Variante à la main
for k in range(1000):
    loss = modele.train_on_batch(entree, sortie)
    print('Erreur :',loss)


# Partie D. Visualisation

sortie_produite = modele.predict(entree)
ZZ = sortie_produite.reshape(Z.shape)  # sortie produite aux bonnes dimensions

# Affichage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('axe x')
ax.set_ylabel('axe y')
ax.set_zlabel('axe z')

# ax.plot_surface(X, Y, Z, color='blue', alpha=0.7)
ax.plot_surface(X, Y, ZZ, color='red', alpha=0.8)


plt.tight_layout()
ax.view_init(20, -45)
# plt.savefig('tf2-deuxvar-fonction-approx.png')
plt.show()




