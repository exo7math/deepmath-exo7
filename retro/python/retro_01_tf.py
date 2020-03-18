import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from keras_facile import *


modele = Sequential()

# Un neurone à 2 entrées 
modele.add(Dense(1, input_dim=2, activation='sigmoid'))

sgd = optimizers.SGD(lr=1)
modele.compile(loss='mean_squared_error', optimizer=sgd)

# poids_a_zeros(modele,0)  # tous les poids à zéros
definir_poids(modele,0,0,[0,1],-2)  # couche, rang, [coeffs], biais 

# carré rouge
# rond bleus 
# à trouver environ y = 1/2x+1, càd a= 1/2, b = -1, c = 1  à un facteur multplicatif près
carres_rouges = [(1,1), (2,0.5), (2,2), (3,1.5), (3,2.75), (4,1), (4,2.5), (4.5,3), (5,1), (5,2.25)]
ronds_bleus   = [(0,3), (1,1.5), (1,4), (1.5,2.5), (2,2.5), (3,3.5), (3.5,3.25), (4,3), (4,4), (5,4)] 


X_train = np.array(carres_rouges+ronds_bleus)
# rouge = 1, bleu = 0
Y_train = np.array( [[1]]*len(carres_rouges) + [[0]]*len(ronds_bleus))

# print(X_train)
# print(Y_train)


# From mpariente https://stackoverflow.com/questions/51140950/
def get_weight_grad(model, inputs, outputs):
    """ Gets gradient of model for given inputs and outputs for all weights"""
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    print("gradient",x,y,sample_weight)
    output_grad = f(x + y + sample_weight)
    return output_grad



print('====================')


# poids_a_zeros(modele,0) 
definir_poids(modele,0,0,[0,1],-2)  # couche, rang, [coeffs], biais 

affiche_poids(modele,0)

# weight_grads = get_weight_grad(modele, X_train, Y_train)
# print(weight_grads)
# 

modele.fit(X_train, Y_train, epochs=1, batch_size=len(X_train), verbose = 1)
affiche_poids(modele,0)

# loss = model.train_on_batch(X_train, Y_train)  # renvoie l'erreur avant l'application du gradient
# print('loss',loss)



# # idem mais une époque à la fois
# permet d'afficher les gradients
# poids_a_zeros(modele,0)  # tous les poids à zéros

# definir_poids(modele,0,0,[1,-1],0)  # couche, rang, [coeffs], biais 
# lr = K.eval(sgd.lr)  # learning rate

# for i in range(10):
#     poids_avant = modele.get_weights()
#     gradient = get_weight_grad(modele, X_train, Y_train)
#     loss = modele.train_on_batch(X_train, Y_train)  # renvoie l'erreur avant l'application du gradient
#     poids_apres = modele.get_weights()
#     poids_calculer_alamain = [poids_avant[i] - lr*gradient[i] for i in range(len(poids_avant))]
#     print("\n==== Epoque numéro",i)

#     print("Poids avant",poids_avant)  
#     print("Gradient",gradient)
#     print("Poids après, par keras",poids_apres)
#     print("Poids calculer à la main",poids_calculer_alamain)
#     print("Erreur",loss)




# Y_train_found = modele.predict(X_train)
# Y_test_found = modele.predict(X_test)
# Affichage
# plt.scatter(X_train, Y_train, s=100,  color='blue')
# plt.scatter(X_train, Y_train_found,  color='red')
# plt.scatter(X_test, Y_test, s=100,  color='cyan')
# plt.scatter(X_test, Y_test_found,  color='green')

# X_liste = np.linspace(0,3,30)
# Y_liste = model.predict(X_liste)
# plt.plot(X_liste,Y_liste)
# plt.show()


# A faire : régression linéaire donnée (x=nb_pair,y).
# Calcul des coeff a,b
# Validation sur un jeu de test ?? (x=nb_impair,y)