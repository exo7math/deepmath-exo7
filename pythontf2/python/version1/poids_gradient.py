import numpy as np
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras import backend as K

modele = Sequential()

# Première couche : 3 neurones (entrée de dimension 2)
modele.add(Dense(3, input_dim=2, activation='relu'))

# Deuxième couche : 1 neurone
modele.add(Dense(1, activation='relu'))

delta = 0.001 

mysgd = optimizers.SGD(learning_rate=delta)
modele.compile(loss='mean_squared_error', optimizer=mysgd)


# Partie A - Les poids
poids_avant = [ np.array([[1.,3.,5.], [2.,4.,6.]]),  # Coeff. couche 1
                np.array([-1.,-2.,-3.]),             # Biais couche 1                
                np.array([[7.], [8.], [9.]]),         # Coeff. couche 2
                np.array([-4.]) ]                    # Biais couche 2

modele.set_weights(poids_avant) # Définis les poids à une valeur sauvegardée

poids_avant = modele.get_weights() # Récupère les poids (ne sert à rien ici !)

print("Poids :\n",poids_avant)
print("=== Première couche ===")
print("Coefficients",poids_avant[0])
print("Biais",poids_avant[1])
print("=== Seconde couche ===")
print("Coefficients",poids_avant[2])
print("Biais",poids_avant[3])


# Partie B - Un apprentissage quelconque

X_train = np.ones((4, 2))
Y_train = np.ones((4, 1))

print(Y_train)

loss = modele.train_on_batch(X_train, Y_train)

poids_apres = modele.get_weights()  # Nouveau poids calculer par tf

modele.set_weights(poids_avant) # Définis les poids à une valeur sauvegardée




# Partie C - Le gradients

#### New 2022 : utilise Gradient Tape

import tensorflow as tf

def compute_gradient(model, X, Y):
    # keep track of our gradients
    with tf.GradientTape() as tape:
        # make a prediction using the model and then calculate the loss
        Y_pred = model(X)
        loss = keras.losses.mean_squared_error(Y, Y_pred)
    # calculate the gradients using our tape 
    grads = tape.gradient(loss, model.trainable_variables)

    return grads

    # opt.apply_gradients(zip(grads, model.trainable_variables))


gradient = compute_gradient(modele, X_train, Y_train)
print("=== Gradient ===")
print( [gradient[i].numpy() for i in range(len(gradient))] )


# # Calculs possibles

# delta = 0.001  # learning rate
N = len(Y_train)  # Il faut diviser par 1/N pour avoir le vrai gradient !!

poids_calcul_alamain = [poids_avant[i] - 1/N*delta*(gradient[i].numpy()) for i in range(len(poids_avant))]


# # Comparaison tensorflow/calculs à la main
print("Poids calculés par tensorflow :\n",poids_apres)
print("Poids calculés à la main :\n",poids_calcul_alamain)
