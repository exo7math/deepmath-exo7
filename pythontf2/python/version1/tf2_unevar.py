import numpy as np
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
# COPIER-COLLER A PARTIR D'ICI


# Partie A. Données

# Fonctions à approcher
def f(x):
    return np.cos(2*x) + x*np.sin(3*x) + x**0.5 - 2

a, b = 0, 5                 # intervalle [a,b]
N = 100                     # taille des données
X = np.linspace(a, b, N)    # abscisses
Y = f(X)                    # ordonnées
X_train = X.reshape(-1,1)
Y_train = Y.reshape(-1,1)

# Partie B. Réseau 

modele = Sequential()

p = 10
modele.add(Dense(p, input_dim=1, activation='tanh'))
modele.add(Dense(p, activation='tanh'))
modele.add(Dense(p, activation='tanh'))
modele.add(Dense(p, activation='tanh'))
modele.add(Dense(1, activation='linear'))


# Partie C. Apprentissage

# Méthode de gradient : descente de gradient classique
weights = modele.get_weights()  # Sauvegarder les poids
mysgd = optimizers.SGD(lr=0.001)
modele.compile(loss='mean_squared_error', optimizer=mysgd)
history_sgd = modele.fit(X_train, Y_train, epochs=4000, batch_size=N)


# Méthode de gradient : descente de gradient classique améliorée
modele.set_weights(weights)  
mysgd = optimizers.SGD(lr=0.001, decay=1e-7, momentum=0.9, nesterov=True)
modele.compile(loss='mean_squared_error', optimizer=mysgd)
history_nesterov = modele.fit(X_train, Y_train, epochs=4000, batch_size=N)

# Méthode de gradient : 'adam'
modele.set_weights(weights)
modele.compile(loss='mean_squared_error', optimizer='adam')
history_adam = modele.fit(X_train, Y_train, epochs=4000, batch_size=N)


# Partie D. Visualisation

# Affichage de la fonction et de son approximation par la dernière méthode
Y_predict = modele.predict(X_train)
plt.plot(X_train, Y_train, color='blue')
plt.plot(X_train, Y_predict, color='red')
plt.show()



# Affichage de l'erreur Nesterov
plt.plot(history_nesterov.history['loss'], color='blue')
# plt.savefig('unevar-fonction-erreur.png')
plt.show()


# Affichage des erreurs
plt.plot(history_sgd.history['loss'], label='gradient classique', color='green')
plt.plot(history_nesterov.history['loss'], label='gradient améliorée', color='blue')
plt.plot(history_adam.history['loss'], label='adam', color='red')
plt.legend()
# plt.savefig('unevar-fonction-leserreurs.png')
plt.show()