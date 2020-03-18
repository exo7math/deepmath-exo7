#!/usr/bin/python3

# Je cherche l'exemple le plus simple ou deux minimums locaux
# insipré par :
# LOCAL MINIMA  IN TRAINING  OF DEEP NETWORKS Grzegorz 
# Swirszcz, Wojciech Marian Czarnecki & Razvan Pascanu

from descente import *
from mpl_toolkits.mplot3d import Axes3D

# Fonction tanh
# sa dérivée tanh'(x) = 1 - tanh(x)**2
def tanh(x):
    E = np.exp(-2*x)
    return (1-E)/(1+E)

# Fonction sigma
def sigma(x):
    return 1/(1+np.exp(-x))

# Fonction heaviside
# définie dans 'descente'
def heaviside(x):
    return np.heaviside(x,1)

def relu(x):
    return np.maximum(x,0)   

# points = [(-3,0), (-2,0),  (0,0.5), (0,0), (2,1), (3,1)]
points = [(-2,1),  (-1,0), (0,0), (1,0), (2,1)]

# Fonction produite par le neurone
def F(x,poids,phi):
    a, b = poids
    S1 = phi(a*x-1)
    # S2 = relu(-S1+b)
    return S1


def E(a,b):
    erreur = 0
    N = len(points)
    for x, y in points:
        erreur += (F(x,(a,b),relu) - y)**2
    return erreur




# Affichage
def graphique_xy():
    # Points
    for x, y in points:
        plt.scatter(x, y, marker='s', color='blue')

    # Fonction
    a, b = 1,-1
    a, b = -1, -1
    # a, b = 1,-2
    # a, b = 0.25, 0.5
    # a, b = 1, -2
    n = 100
    X = np.linspace(-4, 4, n)
    Y = F(X, (a,b), relu)

    plt.plot(X, Y, color='red') 
    plt.tight_layout()
    # plt.savefig('retro_04.png')
    plt.show()
    return


graphique_xy()


# Affichage
def graphique_a():
    n = 100
    X = np.linspace(-2, 2, n)
    Y = E(X,1)

    plt.plot(X, Y, color='red') 
    plt.tight_layout()
    # plt.savefig('retro_04.png')
    plt.show()
    return

graphique_a()

# Affichage
def graphique_ab():
    n = 20
    VX = np.linspace(-2, 2, n)  # a
    VY = np.linspace(-3, 1, n)  # b
    X,Y = np.meshgrid(VX, VY)
    Z = E(X,Y)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel("a axis")
    ax.set_ylabel("b axis")
    # ax.set_zlim(0,2)
    ax.view_init(40, -120)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    # plt.savefig('pythonxy-surface-3.png')
    plt.show()
    return

# graphique_ab()
