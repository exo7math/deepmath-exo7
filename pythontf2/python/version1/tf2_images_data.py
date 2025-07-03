from tensorflow import keras
from tensorflow.keras.datasets import cifar10

# Partie A. Données

(X_train_data, Y_train_data), (X_test_data, Y_test_data) = cifar10.load_data()

num_classes = 10
labels =  ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

Y_train = keras.utils.to_categorical(Y_train_data, num_classes)
X_train = X_train_data.reshape(50000,32*32*3)
X_train = X_train.astype('float32')
X_train = X_train/255

# Partie A bis. Afficher des images
import matplotlib.pyplot as plt

def affiche_images(debut):
    plt.axis('off')
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.title(labels[Y_train_data[i+debut][0]])
        plt.imshow(X_train_data[i], interpolation='nearest')
    plt.tight_layout()
    # plt.savefig('tf2-images-train.png')    
    plt.show()

    return

affiche_images(0)  # affiche des images à partir de 0
