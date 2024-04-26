
# Plongement des chiffres MNIST par réseaux de neurones

import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout



import matplotlib.pyplot as plt


### Partie A - Création des données

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(X_train_data, Y_train_data), (X_test_data, Y_test_data) = mnist.load_data()

N = X_train_data.shape[0]  # 60 000 données

# print(X_train_data[0].shape)
# print(X_train_data[0])


X_train = np.reshape(X_train_data, (N,28,28,1))
X_test = np.reshape(X_test_data, (X_test_data.shape[0],28,28,1))

# X_train = p.reshape(X_train_data,(N,784))  # vecteur image
# X_test = np.reshape(X_test_data,(X_test_data.shape[0],784))

X_train = X_train/255  # normalisation
X_test = X_test/255

# print(X_train[0].shape)
# print(X_test[0].shape)

Y_train = to_categorical(Y_train_data, num_classes=10)
Y_test = to_categorical(Y_test_data, num_classes=10)

# print(Y_train[0])
# print(Y_train_data[0])

### Partie B - Réseau de neurones

modele = Sequential()

# Première couche de convolution : 16 neurones, convolution 3x3, activation relu
modele.add(Conv2D(16, kernel_size=3, padding='same', activation='relu', input_shape=(28,28,1)))
modele.add(MaxPooling2D(pool_size=(2, 2)))

# Deuxième couche de convolution : 8 neurones
modele.add(Conv2D(8, kernel_size=3, padding='same', activation='relu'))
modele.add(MaxPooling2D(pool_size=(2, 2)))

# 
# modele.add(Conv2D(8, kernel_size=3, activation='relu'))

# Aplatissage 
modele.add(Flatten())

# Avant dernière couche (cachée) : 50 neurones
modele.add(Dense(50, activation='relu'))


# Couche de sortie : 1O neurone
modele.add(Dense(10, activation='softmax'))

# Descente de gradient
modele.compile(loss='categorical_crossentropy', 
              optimizer='adam',  
              metrics=['accuracy'])

print(modele.summary())


# Calcul des poids
modele.fit(X_train, Y_train, batch_size=32, epochs=5, verbose=1)


### Partie C - Résultats

score = modele.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# Dernière couche avant softmax et la sortie
# Fonction qui prend en entrée une image et renvoie la sortie de la couche (dans un espace de dimension 392)
func = K.function([modele.input], [modele.layers[-2].output])


### Partie D - Visualisation

# Analyse en composantes principales
# cad meilleure projection de l'espace des vecteurs sur un espace de dimension 2
# https://stackoverflow.com/questions/36771525/python-pca-projection-into-lower-dimensional-space
def pca(X, k=2):
    # Centrer les données
    Xmean = X.mean(axis=0)
    XX = X - Xmean
    # Matrice de covariance
    C = np.dot(XX.T, XX) / (XX.shape[0] - 1)
    # Décomposition en valeurs propres
    d, u = np.linalg.eigh(C)
    # Tri des valeurs propres
    idx = np.argsort(d)[::-1]
    # Tri des vecteurs propres
    u = u[:, idx]
    # Projection sur les k premiers vecteurs propres
    U = u[:, :k]
    return np.dot(XX, U), Xmean, u[:, :k], d[idx]


def plongement_chiffres():
    nbsamples = 150   # nb de chiffres à afficher
    YY = func(X_test[0:nbsamples])
    # print(type(YY[0]))

    ten_colors = ['lightblue', 'orange', 'purple', 'green', 'red', 'lime', 'pink', 'gray', 'olive', 'cyan']

    # Affichage 2D
    Z, _, _, _ = pca(YY[0], k=2)

    for i in range(nbsamples):
        plt.scatter(Z[i, 0], Z[i, 1], c=ten_colors[Y_test_data[i]])
        plt.annotate(str(Y_test_data[i]), (Z[i, 0], Z[i, 1]), ha='center', weight='bold', fontsize=10)

    for k in range(10):
        plt.scatter([], [], c=ten_colors[k], label=str(k))

    plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    # plt.savefig("mnist-embed-1.png", dpi=600)
    plt.show()


    # Affichage 3D
    Z, _, _, _ = pca(YY[0], k=3)

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = plt.subplot(111, projection="3d")

    for i in range(nbsamples):
        ax.scatter(Z[i, 0], Z[i, 1], Z[i, 2], c=ten_colors[Y_test_data[i]])

    # for k in range(10):
    #     ax.scatter([], [], [], c=ten_colors[k], label=str(k))

    # plt.legend()
    # plt.axis("off")
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    ax.view_init(azim=-40, elev=20)
    # plt.xticks([])
    # plt.yticks([])
    # plt.zticks([])
    # plt.tight_layout()
    # plt.savefig("mnist-embed-2.png", dpi=600)
    plt.show()

    return


plongement_chiffres()
