#!/usr/bin/python3

# Je cherche l'exemple le plus simple avec deux minimums locaux
# insipré par :
# LOCAL MINIMA  IN TRAINING  OF DEEP NETWORKS Grzegorz 
# Swirszcz, Wojciech Marian Czarnecki & Razvan Pascanu

from descente import *


def relu(x):
    return np.maximum(x,0)   

# points = [(-3,0), (-2,0),  (0,0.5), (0,0), (2,1), (3,1)]
# points = [(-2,1),  (-1,0), (0,0), (1,0), (2,1)]
# points = [(-4,1),  (-1,0), (0,0), (1,0), (2,1)]
points = [(-2,1),  (-1,0), (0,0), (1,0), (3,2)]


# Fonction produite par le neurone
def F(x,a):
    return relu(a*x-1)


def E(a):
    erreur = 0
    N = len(points)
    for x, y in points:
        erreur += (F(x,(a)) - y)**2
    return erreur




# Affichage
def graphique_xy():
    # Points
    for x, y in points:
        plt.scatter(x, y, marker='s', color='blue')

    # Fonction
    a = -1  # poids
    a = 1
    a = 2
 
    n = 100
    X = np.linspace(-4, 4, n)
    Y = F(X, a)

    plt.plot(X, Y, color='red') 
    plt.xlim(-4,4)
    plt.ylim(-0.5,3)
    plt.tight_layout()
    # plt.savefig('retro_04_d.png')
    plt.show()
    return


graphique_xy()


# Affichage
def graphique_a():
    n = 100
    X = np.linspace(-1.7, 1.8, n)
    Y = E(X)

    # plt.xlim(-0.5,1.5)
    plt.xlabel("coefficient a")
    plt.ylabel("erreur E(a)")
    plt.ylim(0,8)
    plt.plot(X, Y, color='red') 
    plt.tight_layout()
    # plt.savefig('retro_04_e.png')
    plt.show()
    return

graphique_a()


