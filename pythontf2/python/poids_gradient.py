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

mysgd = optimizers.SGD(lr=0.001)
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
loss = modele.train_on_batch(X_train, Y_train)

poids_apres = modele.get_weights()  # Nouveau poids calculer par tf


modele.set_weights(poids_avant) # Définis les poids à une valeur sauvegardée


# Partie C - Le gradients

# Références 'mpariente' https://stackoverflow.com/questions/51140950/
def get_weights_grad(model, inputs, outputs):
    """ Gets gradient of model for given inputs and outputs for all weights"""
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    return output_grad


gradient = get_weights_grad(modele, X_train, Y_train)
print("Gradient :\n", gradient)

# Calculs possibles
delta = 0.001  # learning rate
poids_calculer_alamain = [poids_avant[i] - delta*gradient[i] for i in range(len(poids_avant))]

# Comparaison tensorflow/calculs à la main
print("Poids calculer par tensorflow :\n",poids_apres)
print("Poids calculer à la main :\n",poids_calculer_alamain)
