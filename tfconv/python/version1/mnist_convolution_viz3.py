#!/usr/bin/python3

import numpy as np
from tensorflow import keras
# from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten


from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt

### Partie A - Création des données

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(X_train_data, Y_train_data), (X_test_data, Y_test_data) = mnist.load_data()

N = X_train_data.shape[0]  # 60 000 données

X_train = np.reshape(X_train_data, (N,28,28,1))
X_test = np.reshape(X_test_data, (X_test_data.shape[0],28,28,1))

X_train = X_train/255  # normalisation
X_test = X_test/255

Y_train = to_categorical(Y_train_data, num_classes=10)
Y_test = to_categorical(Y_test_data, num_classes=10)


### Partie B - Réseau de neurones

# Il faut d'abord executer 'mnist_convolution_viz1.py'
# puis on récupère le modèle :
model = load_model('modele_mnist_viz.h5')

my_layer = 3   # layer nb to be inspected

# redefine model to output right after our hidden layer
model = Model(inputs=model.inputs, outputs=model.layers[my_layer-1].output)

num_filters = model.layers[my_layer].output_shape[3]
print("Nom de la sous-couche :",model.layers[my_layer].name)
print("Nb de sous-couches :",num_filters)

# print(model.summary())

### Partie D - Visualisation couches intermédiaures
# https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/
# retrieve weights from the second hidden layer

# One random image : (check #56='4', #57='1', #58='9')

# fig = plt.figure(figsize=(5,5))
# plt.imshow(img[0,:,:,0],cmap="Greys")
# plt.axis('off')
# plt.show()

# Output of the image by the layer
# feature_maps = model.predict(img)

num_filt = 15  # un filtre au pif (attention décalage d'un rang avec affichage #filtre)
num_img = [69,57,77,63,56,59,81,64,84,62]

# plot all figures for only one filters
Nx, Ny = 2, 5

fig = plt.figure(figsize = (10,5))
for n in range(len(num_img)):
        img = X_test[num_img[n]].reshape(1,28,28,1)
        feature_maps = model.predict(img)
        # specify subplot and turn of axis
        ax = plt.subplot(Nx, Ny, n+1)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.title('Chiffre %d' % n )
        plt.imshow(feature_maps[0, :, :, num_filt], cmap='hot')  # 'YlOrRd'

# show the figure
fig.suptitle('Couche %d - Filtre %d' % (my_layer,num_filt+1) )
plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.95, hspace=1.0,wspace=0.5)
plt.tight_layout()
# plt.savefig('tfconv-viz3-c3-f16.png')  # c = #couche, f = #filtre
plt.show()