# Linguistique et statistique
# Fonction softmax et température


import numpy as np
import matplotlib.pyplot as plt


#### Softmax et température ####

# Fonction softmax
def softmax(X):
    """ Fonction softmax classique """
    som = sum([np.exp(x) for x in X])
    res = [np.exp(x)/som for x in X]   
    return res


# Fonction softmax avec température
def softmaxT(X, T):
    """ Fonction softmax avec température """
    som = sum([np.exp(x/T) for x in X])
    res = [np.exp(x/T)/som for x in X]   
    return res


# Fonction choix aléatoire avec poids
def aleatoire_poids(P):
    """ Fonction qui choisit aléatoirement un indice en fonction des poids
     on part du principe que la somme des poids fait 1 """
    som = sum(P)
    P = [p/som for p in P]   # pour être sûr que la somme des poids fasse 1
    r = np.random.rand()     # nombre aléatoire entre 0 et 1
    sp = 0
    for i in range(len(P)):
        sp += P[i]
        if sp > r:
            return i
    return len(P)-1   # juste au cas où pb d'arrondi à 0.99...


def exemple0():
    print(softmax([1, 2, 3]))
    print(softmaxT([1, 2, 3], 2.0))
    for T in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
        print(T, softmaxT([1, 2, 3], T))
    print(softmaxT([1, 2, 3], 0.5))
    print(aleatoire_poids([0.1, 0.2, 0.7]))
    return

def exemple1():
    # Graphe de la fonction softmax pour poids (x,1-x)
    def mysoft(x):
        return np.exp(x)/(np.exp(x)+np.exp(1-x))

    def mysoftT(x, T):
        return np.exp(x/T)/(np.exp(x/T)+np.exp((1-x)/T))

    plt.figure(figsize=(15, 5))
    X = np.linspace(-3, 4, 300)
    Y = mysoft(X)
    plt.plot(X, Y)
    # YY = mysoftT(X, 0.3)
    # plt.plot(X, YY)
    # YY = mysoftT(X, 3)
    # plt.plot(X, YY)

    # for T in [0.1, 0.5, 1.0, 2.0, 4.0]:
    #     YY = mysoftT(X, T)
    #     plt.plot(X, YY, label="T="+str(T))

    # plt.legend()
    # plt.tight_layout()
    # # # plt.savefig("softmax-01.png", dpi=600)
    # plt.show()  
    return 




def exemple2():
    """ Fonctions softmax et softmaxT avec température 
    sous forme de diagramme en barres """
    # np.random.seed(2)
    # X = np.random.rand(7)
    k = 6
    X = [2,5,25,3,14,4]
    print(sum(X))
    X = [x/sum(X) for x in X]   # on se ramène à une somme 1

    plt.figure(figsize=(15, 5))
    mycolor = ["red", "green", "blue", "brown", "gray", "purple", "orange"]

    # plt.subplot(1, 1, 1)
    # plt.bar(range(1,k+1), X, color=mycolor)
    # plt.ylim(0, 1)
    # plt.title("X")
    # plt.tight_layout()
    # # plt.savefig("softmax-03.png", dpi=600)
    # plt.show()

    for i, T in enumerate([0.1, 0.2, 1, 5]):
        plt.subplot(2, 2, i+1)
        Y = softmaxT(X, T)
        plt.bar(range(1,k+1), Y, color=mycolor)
        plt.ylim(0, 1)
        plt.title("T = "+str(T))
   
    plt.tight_layout()
    # plt.savefig("softmax-04.png", dpi=600)
    plt.show()


    return

exemple2()







