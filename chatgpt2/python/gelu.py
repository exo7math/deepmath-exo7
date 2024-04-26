# La fonction d'activation GELU

import numpy as np
import matplotlib.pyplot as plt
import math

### Partie A : loi normale et fonction de répartition


def graphe1():
    """ Graphe de la loi normale """
    plt.figure(figsize=(10, 5))
    X = np.linspace(-5, 5, 100)
    Y = np.exp(-X**2/2) / np.sqrt(2*np.pi)
    plt.plot(X, Y, color="blue", linewidth=2)
    plt.tight_layout()
    # plt.savefig('gelu01.png', dpi=600)
    plt.show()

    return

def Phi(x):
    return (1 + math.erf(x / math.sqrt(2))) / 2

def graphe2():
    """ Graphe de la fonction de répartition """
    plt.figure(figsize=(10, 5))
    X = np.linspace(-5, 5, 100)
    Y = np.vectorize(Phi)(X)    
    plt.plot(X, Y, color="orange", linewidth=2)
    plt.tight_layout()
    # plt.savefig('gelu02.png', dpi=600)
    plt.show()

    return

# graphe1()
# graphe2()

### Partie B : GELU

def gelu(x):
    return x/2 * (1 + math.erf(x / math.sqrt(2)))

def relu(x):
    return np.maximum(0, x)

def graphe3():
    """ Graphe de la fonction d'activation GELU """
    plt.figure(figsize=(10, 5))
    X = np.linspace(-4, 3, 100)
    Y = np.vectorize(gelu)(X)
    plt.plot(X, Y, color="blue", linewidth=2)
    plt.tight_layout()
    # plt.savefig('gelu03.png', dpi=600)
    plt.show()

    return

def graphe4():
    """ GELU vs ReLU """
    plt.figure(figsize=(10, 5))
    X = np.linspace(-4, 3, 100)
    Y1 = np.vectorize(gelu)(X)
    Y2 = relu(X)
    plt.plot(X, Y2, label="ReLU", color="orange", linewidth=2)
    plt.plot(X, Y1, label="GELU", color="blue", linewidth=2)
    plt.legend()
    plt.tight_layout()
    # plt.savefig('gelu04.png', dpi=600)
    plt.show()

    return

# graphe3()
# graphe4()


### Partie C : approximation de GELU

def gelu_th(x):
    """" Approximation de GELU à l'aide de la tangente hyperbolique """
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def sigma(x):
    """ Fonction sigmoïde """
    return 1 / (1 + np.exp(-x))

def gelu_sigma(x):
    """" Approximation de GELU à l'aide de la sigmoïde """
    return x*sigma(1.702*x)


def graphe5():
    """ Approximations de GELU """
    plt.figure(figsize=(10, 5))
    X = np.linspace(-3, 2, 100)
    Y1 = np.vectorize(gelu)(X)
    Y2 = np.vectorize(gelu_th)(X)
    Y3 = np.vectorize(gelu_sigma)(X)
    plt.plot(X, Y1, label="GELU", color="blue", linewidth=2)
    plt.plot(X, Y2, label="GELU_th", color="orange", linewidth=2)
    plt.plot(X, Y3, label="GELU_sigma", color="green", linewidth=2)
    plt.legend()
    plt.tight_layout()
    # plt.savefig('gelu05.png', dpi=600)
    plt.show()

    return


def graphe6():
    """ Erreur de ces approximations """
    plt.figure(figsize=(10, 5))
    X = np.linspace(-4, 4, 100)
    Y1 = np.vectorize(gelu)(X)
    Y2 = np.vectorize(gelu_th)(X)
    Y3 = np.vectorize(gelu_sigma)(X)
    plt.plot(X, Y1 - Y2, label="GELU - GELU_th", color="orange", linewidth=2)
    plt.plot(X, Y1 - Y3, label="GELU - GELU_sigma", color="green", linewidth=2)
    plt.legend()
    plt.tight_layout()
    # plt.savefig('gelu06.png', dpi=600)
    plt.show()

    return

# graphe5()
# graphe6()

    
