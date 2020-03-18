#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

# Exemples de fonctions avec leur gradient
# f1 = x**2 + 3*y**2
# g1 = [2*x, 6*y]

# f2 = x**2 + (y-x**2)**2
# g2 = [2*x-4*x*(y-x**2), 2*(y-x**2)]


# fonction de 2 variables
def f(x, y):
    fonc = x**2 + (x-y**2)**2
    return fonc
    
# gradient calculer par nos soins
def gradient_f(x, y):
    grad = [2*x+2*(x-y**2), -2*y*(x-y**2)]
    return np.array(grad)




def descente(X0, delta=0.1, nmax=10):
    liste_X = [X0]
    liste_grad = []
    X = X0
    for i in range(nmax):
        gradient = gradient_f(*X)
        X = X - delta*gradient
        liste_X.append(X)
        liste_grad.append(gradient)
    return liste_X, liste_grad


# Test
print("--- Descente de gradient ---")
X0 = np.array([2, 1])
liste_X, liste_grad = descente(X0, delta=0.1)
print("Points :", liste_X)
print("Gradients :", liste_grad)


def affiche_descente(X0, delta=0.1, nmax=10):
    liste_X, liste_grad = descente(X0, delta=delta, nmax=nmax)
    print("Delta",delta)
    print("Nombre d'itérations", nmax)
    print("Point initial", X0)
    for i in range(len(liste_X)-1):    # flèches
        print("Point :", *liste_X[i], "Gradient ", *liste_grad[i])
    print("Dernier point :", *liste_X[-1])
    return


# Test
# X0 = np.array([2, 2])
# affiche_descente(X0, delta=0.1)


def graphique_descente(X0, delta=0.1, nmax=10):
    # 1. Points et gradients
    liste_X, liste_grad = descente(X0, delta=delta, nmax=nmax)
    for x, y in liste_X:    # points
        plt.scatter(x, y, color='red')

    for i in range(len(liste_X)-1):    # flèches
        plt.arrow(*liste_X[i], *(-delta*liste_grad[i]), linewidth=2, color='orange', length_includes_head=True, head_width=0.1, head_length=0.15)

    # 2. lignes de niveaux
    xmin, xmax = -3.0, 3.0
    ymin, ymax = -3.0, 3.0
    num = 40
    VX = np.linspace(xmin, xmax, num)
    VY = np.linspace(ymin, ymax, num)

    X, Y = np.meshgrid(VX, VY)
    Z = f(X, Y)

    # 3. affichage
    plt.contour(X, Y, Z, 30, colors='black')
    plt.scatter(0,0, color='blue')  # minimum
    # plt.colorbar();
    plt.axis('equal')
    plt.tight_layout()
    # plt.savefig('descente.png')
    plt.show()
    return


# Test
X0 = np.array([2, 2])
mon_delta = 0.3
affiche_descente(X0, delta=mon_delta)
graphique_descente(X0, delta=mon_delta)