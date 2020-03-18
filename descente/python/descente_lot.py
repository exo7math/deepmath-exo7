#!/usr/bin/python3

# Descente de gradient 

import numpy as np
import matplotlib.pyplot as plt


def descente_lot(f, grad_fi, points, K, X0, delta=0.1, epoques=3):
    liste_X = [X0]
    liste_grad = []
    X = X0
    N = len(points)

    for e in range(epoques):   # pour chaque époques
        j = 0                  # numérotation dans le lot
        for i in range(N):     # pour chaque données
            xi, yi = points[i]
            if j == 0:
                gradient_lot = grad_fi(*X, xi, yi)
            else:
                gradient_lot = gradient_lot + grad_fi(*X, xi, yi)
            
            j = j+1

            if j == K:   # fin de lot
                X = X - delta*gradient_lot
                liste_X.append(X)
                liste_grad.append(gradient_lot)
                j = 0    # remise à zéro pour lot suivant


    return liste_X, liste_grad


def affiche_descente_lot(f, grad_fi, points, K, X0, delta=0.1, epoques=3):
    liste_X, liste_grad = descente_lot(f, grad_fi, points, K, X0, delta=delta, epoques=epoques)
    print("Delta",delta)
    print("Nombre d'époques", epoques)
    print("Taille des lots", K)    
    print("Point initial", X0)
    for i in range(len(liste_X)-1):    # flèches
        print("--- Etape",i)
        print("Point :", *liste_X[i])
        print("Gradient ", *liste_grad[i])
        print("Valeur de la fonction ", f(*liste_X[i]))
    print("Dernier point :", *liste_X[-1])
    print("Dernière valeur de la fonction ", f(*liste_X[-1]))
    return


def graphique_descente_lot(f, grad_fi, points, K, X0, delta=0.1, epoques=3, zone = (-3.0,3.0,-3.0,3.0)):
    # 1. Points et gradients
    liste_X, liste_grad = descente_lot(f, grad_fi, points, K, X0, delta=delta, epoques=epoques)
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
    plt.savefig('descente_lot.png')
    plt.show()
    return