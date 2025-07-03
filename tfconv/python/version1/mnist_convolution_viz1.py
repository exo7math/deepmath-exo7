#!/usr/bin/python3

import numpy as np
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

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

print(X_train[0].shape)
print(X_test[0].shape)

Y_train = to_categorical(Y_train_data, num_classes=10)
Y_test = to_categorical(Y_test_data, num_classes=10)

print(Y_train[0])
print(Y_train_data[0])

### Partie B - Réseau de neurones

modele = Sequential()

# Première couche de convolution : 16 neurones, convolution 3x3, activation relu
modele.add(Conv2D(16, kernel_size=3, padding='same', activation='relu', input_shape=(28,28,1)))

# Deuxième couche de convolution : 16 neurones
modele.add(Conv2D(16, kernel_size=3, padding='same', activation='relu'))

# Troisième couche de convolution : 16 neurones
# modele.add(Conv2D(16, kernel_size=3, padding='same', activation='relu'))

modele.add(MaxPooling2D(pool_size=(2, 2)))

# Aplatissage 
modele.add(Flatten())

# Couche de sortie : 1O neurone
modele.add(Dense(10, activation='softmax'))

# Descente de gradient
modele.compile(loss='categorical_crossentropy', 
              optimizer='adam',  
              metrics=['accuracy'])

print(modele.summary())


# Calcul des poids
modele.fit(X_train, Y_train, batch_size=32, epochs=3, verbose=1)

modele.save(filepath='modele_mnist_viz.h5')

### Partie C - Résultats

score = modele.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

