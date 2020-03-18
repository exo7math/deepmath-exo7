#!/usr/bin/python3

# Descente de gradient 

import numpy as np
import matplotlib.pyplot as plt


def descente_stochastique(f, grad_fi, points, X0, delta=0.1, nmax=10):
    liste_X = [X0]
    liste_grad = []
    X = X0
    N = len(points)
    for i in range(nmax):
        xi, yi = points[i%N]
        gradienti = grad_fi(*X,xi,yi)
        X = X - delta*gradienti
        liste_X.append(X)
        liste_grad.append(gradienti)
    return liste_X, liste_grad


def affiche_descente_stochastique(f, grad_fi, points, X0, delta=0.1, nmax=10):
    liste_X, liste_grad = descente_stochastique(f, grad_fi, points, X0, delta=delta, nmax=nmax)
    print("Delta",delta)
    print("Nombre d'itérations", nmax)
    print("Point initial", X0)
    for i in range(len(liste_X)-1):    # flèches
        print("--- Etape",i)
        print("Point :", *liste_X[i])
        print("Gradient ", *liste_grad[i])
        print("Valeur de la fonction ", f(*liste_X[i]))
    print("Dernier point :", *liste_X[-1])
    print("Dernière valeur de la fonction ", f(*liste_X[-1]))
    return


def graphique_descente_stochastique(f, grad_fi, points, X0, delta=0.1, nmax=10, zone = (-3.0,3.0,-3.0,3.0)):
    # 1. Points et gradients
    liste_X, liste_grad = descente_stochastique(f, grad_fi, points, X0, delta=delta, nmax=nmax)
    for x, y in liste_X:    # points
        plt.scatter(x, y, color='red')

    for i in range(len(liste_X)-1):    # flèches
        plt.arrow(*liste_X[i], *(-delta*liste_grad[i]), linewidth=2, color='green', length_includes_head=True, head_width=0.05, head_length=0.1)

    # 2. lignes de niveaux
    xmin, xmax, ymin, ymax = zone
    num = 40
    VX = np.linspace(xmin, xmax, num)
    VY = np.linspace(ymin, ymax, num)

    X, Y = np.meshgrid(VX, VY)
    Z = f(X, Y)

    # 3. affichage
    plt.contour(X, Y, Z, 30, colors='black')
    plt.scatter(-3,2, color='blue')  # minimum
    # plt.colorbar();
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('stochastique.png')
    plt.show()
    return