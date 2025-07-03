import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from keras_facile import *


modele = Sequential()

# Réseau : 
# 1 entrée x, 
# première couche : 2 neurones
# seconde couche : 1 neurone
# activation = sigma

# Première couche : 2 neurones (entrée de dimension 1)
modele.add(Dense(2, input_dim=1, activation='sigmoid'))

# Seconde et dernière couche : 1 neurone
modele.add(Dense(1, activation='sigmoid'))

mysgd = optimizers.SGD(lr=1)
modele.compile(loss='mean_squared_error', optimizer=mysgd)

# poids_a_zeros(modele,0)  # couche 0, tous les poids à zéros
# poids_a_zeros(modele,1)  # couche 0, tous les poids à zéros
definir_poids(modele,0,0,[1/3],-1)  # couche, rang, [coeffs], biais
definir_poids(modele,0,1,[-1/5],1)  # couche, rang, [coeffs], biais
definir_poids(modele,1,0,[1,1],-2)  # couche, rang, [coeffs], biais

definir_poids(modele,0,0,[0],1)  # couche, rang, [coeffs], biais
definir_poids(modele,0,1,[0],-1)  # couche, rang, [coeffs], biais
definir_poids(modele,1,0,[1,1],-1)  # couche, rang, [coeffs], biais

affiche_poids(modele,0)
affiche_poids(modele,1)

# carré rouge +1
# rond bleus 0
# fonction vaut 1 sur [3,5] et 0  sur [0,2] et [6,10] ailleurs
carres_rouges = list(np.linspace(3,5,10))
ronds_bleus   = list(np.linspace(0,2,10)) + list(np.linspace(6,8,10))

X_train = np.array(carres_rouges+ronds_bleus)
# rouge = 1, bleu = 0
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


# modele.fit(X_train, Y_train, epochs=10000, batch_size=len(X_train), verbose = 1)
affiche_poids(modele,0)
affiche_poids(modele,1)



# loss = model.train_on_batch(X_train, Y_train)  # renvoie l'erreur avant l'application du gradient
# print('loss',loss)


# # idem mais une époque à la fois
# permet d'afficher les gradients
# poids_a_zeros(modele,0)  # tous les poids à zéros
# poids_a_zeros(modele,0)  # couche 0, tous les poids à zéros
# poids_a_zeros(modele,1)  # couche 0, tous les poids à zéros
lr = K.eval(mysgd.lr)  # learning rate

# for i in range(1):
#     poids_avant = modele.get_weights()
#     gradient = get_weight_grad(modele, X_train, Y_train)
#     loss = modele.train_on_batch(X_train, Y_train)  # renvoie l'erreur avant l'application du gradient
#     poids_apres = modele.get_weights()
#     poids_calculer_alamain = [poids_avant[i] - lr*gradient[i] for i in range(len(poids_avant))]
#     print("\n==== Epoque numéro",i)

#     print("Poids avant",poids_avant)  
#     print("Gradient",gradient)
#     print("Poids après, par keras",poids_apres)
#     # print("Poids calculer à la main",poids_calculer_alamain)
#     print("Erreur",loss)

# print('========= Erreurs ===========')
liste_erreur = []
for i in range(5001):
    loss = modele.train_on_batch(X_train, Y_train)  # renvoie l'erreur avant l'application du gradient
    liste_erreur.append(loss)
    print('loss',loss)
    modele.evaluate(X_train, Y_train)
print("Liste des erreurs", liste_erreur)


poids = modele.get_weights()
liste_poids = [ poids[0][0][0], poids[1][0], 
                poids[0][0][1], poids[1][1],
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