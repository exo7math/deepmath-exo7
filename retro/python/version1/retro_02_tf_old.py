import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from keras_facile import *


modele = Sequential()

# Réseau : 
# 2 entrées (x,y), 
# première couche : 2 neurones
# seconde couche : 1 neurone
# activation = tangente hyperbolique

# Première couche : 2 neurones (entrée de dimension 2)
modele.add(Dense(2, input_dim=2, activation='sigmoid'))

# Seconde et dernière couche : 1 neurone
modele.add(Dense(1, activation='sigmoid'))

sgd = optimizers.SGD(lr=1)
modele.compile(loss='mean_squared_error', optimizer=sgd)

# poids_a_zeros(modele,0)  # couche 0, tous les poids à zéros
# poids_a_zeros(modele,1)  # couche 0, tous les poids à zéros
definir_poids(modele,0,0,[0,0],1)  # couche, rang, [coeffs], biais
definir_poids(modele,0,1,[1,1],0)  # couche, rang, [coeffs], biais
definir_poids(modele,1,0,[-1,-1],0)  # couche, rang, [coeffs], biais

affiche_poids(modele,0)
affiche_poids(modele,1)

# carré rouge +1
# rond bleus -1
# à trouver : fonction XOR : +1 pour (x>0, y>0) et pour (x<0, y<0) ; -1 sur (x>0, y<0) et (x<0, y>0)
carres_rouges = [(0.0,0.4), (0.1,0.7), (0.2,0.2), (0.2,1.0), (0.4,0.6), (0.6,0.0), (0.6,0.4), (0.8,0.9), (1.0,0.1), (1.0,0.5)]
ronds_bleus   = [(0.1,-0.3), (0.1,-0.7), (0.2,-0.1), (0.3,-0.8), (0.6,-0.1), (0.7,-0.3), (0.8,-0.2), (0.8,-1.0), (0.9,-0.9), (1.0,-0.1) ] 
carres_rouges += [(-x,y) for x,y in ronds_bleus]  # symétriques
ronds_bleus  += [(-x,y) for x,y in carres_rouges]  # symétriques


X_train = np.array(carres_rouges+ronds_bleus)
# rouge = 1, bleu = -1
Y_train = np.array( [[1]]*len(carres_rouges) + [[0]]*len(ronds_bleus) )

# print(X_train)
# print(Y_train)


# From mpariente https://stackoverflow.com/questions/51140950/
def get_weight_grad(model, inputs, outputs):
    """ Gets gradient of model for given inputs and outputs for all weights"""
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    return output_grad



print('====================')


modele.fit(X_train, Y_train, epochs=50000, batch_size=len(X_train), verbose = 1)
affiche_poids(modele,0)
affiche_poids(modele,1)



# loss = model.train_on_batch(X_train, Y_train)  # renvoie l'erreur avant l'application du gradient
# print('loss',loss)




# # idem mais une époque à la fois
# permet d'afficher les gradients
# poids_a_zeros(modele,0)  # tous les poids à zéros
# poids_a_zeros(modele,0)  # couche 0, tous les poids à zéros
# poids_a_zeros(modele,1)  # couche 0, tous les poids à zéros
lr = K.eval(sgd.lr)  # learning rate

for i in range(-1):
    poids_avant = modele.get_weights()
    gradient = get_weight_grad(modele, X_train, Y_train)
    loss = modele.train_on_batch(X_train, Y_train)  # renvoie l'erreur avant l'application du gradient
    poids_apres = modele.get_weights()
    poids_calculer_alamain = [poids_avant[i] - lr*gradient[i] for i in range(len(poids_avant))]
    print("\n==== Epoque numéro",i)

    print("Poids avant",poids_avant)  
    print("Gradient",gradient)
    print("Poids après, par keras",poids_apres)
    # print("Poids calculer à la main",poids_calculer_alamain)
    print("Erreur",loss)


poids = modele.get_weights()
liste_poids = [ poids[0][0][0], poids[0][1][0], poids[1][0], 
                poids[0][0][1], poids[0][1][1], poids[1][1],
                poids[2][0][0], poids[2][1][0], poids[3][0] ]
print("poids\n",liste_poids)

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