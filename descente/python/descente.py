#!/usr/bin/python3

# Descente de gradient 

import numpy as np
import matplotlib.pyplot as plt


def descente(f, grad_f, X0, delta=0.1, nmax=10):
    liste_X = [X0]
    liste_grad = []
    X = X0
    for i in range(nmax):
        gradient = grad_f(*X)
        X = X - delta*gradient
        liste_X.append(X)
        liste_grad.append(gradient)
    return liste_X, liste_grad


def affiche_descente(f, grad_f, X0, delta=0.1, nmax=10):
    liste_X, liste_grad = descente(f, grad_f, X0, delta=delta, nmax=nmax)
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


def graphique_descente_2var_2d(f, grad_f, X0, delta=0.1, nmax=10, zone = (-3.0,3.0,-3.0,3.0)):
    # 1. Points et gradients
    liste_X, liste_grad = descente(f, grad_f, X0, delta=delta, nmax=nmax)

    for x, y in liste_X:    # points
        plt.scatter(x, y, color='red')

    for i in range(len(liste_X)-1):    # flèches
        plt.arrow(*liste_X[i], *(-delta*liste_grad[i]), linewidth=2, color='orange', length_includes_head=True, head_width=0.05, head_length=0.1)

    # 2. lignes de niveaux
    xmin, xmax, ymin, ymax = zone
    num = 40
    VX = np.linspace(xmin, xmax, num)
    VY = np.linspace(ymin, ymax, num)

    X, Y = np.meshgrid(VX, VY)
    Z = f(X, Y)

    # 3. affichage
    # pour intro 
    niveaux = [1.64, 2.95, 8.22]
    # plt.contour(X, Y, Z, niveaux, colors='black')
    plt.contour(X, Y, Z, 30, colors='black')

    plt.scatter(0,0, color='blue')  # minimum
    # plt.colorbar();
    plt.axis('equal')
    plt.tight_layout()
    # plt.savefig('descente_intro_06.png')
    plt.show()
    return

def graphique_regression(f, grad_f, X0, points, delta=0.1, nmax=10, zone = (-3.0,3.0,-3.0,3.0)):
    # 1. Droites
    liste_X, liste_grad = descente(f, grad_f, X0, delta=delta, nmax=nmax)

    num = 40
    xmin, xmax, ymin, ymax = zone
    VX = np.linspace(xmin, xmax, num)

    # Droite initiale
    a, b = X0
    VY = a*VX + b
    plt.plot(VX, VY, linewidth=2, color='red')
    plt.text(6.5, a*6+b, 'k = 0')

    # Droite finale
    a, b = 0.78, -2.46
    VY = a*VX + b
    plt.plot(VX, VY, linewidth=2, color='blue')    
    # plt.text(6.5, a*6+b, 'y = 0.5 x + 3')

    for i in range(1, len(liste_X)):  
        a, b = liste_X[i]
        VY = a*VX + b
        plt.plot(VX, VY, linewidth=2, color='orange')
        plt.text(6.5, a*6+b, 'k = '+str(i))
 
    for x,y  in points:    # points
        plt.scatter(x, y, color='black')
        
    # 3. affichage
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('descente.png')
    plt.show()
    return



def graphique_descente_1var(f, grad_f, X0, delta=0.1, nmax=10):
    xmin, xmax = -1.5, 2.5

    if (nmax==11) and (delta!=0.2):
        xmin, xmax = -3, 3
        plt.axis([xmin,xmax,-1,11])
    
        
    num = 100
    VX = np.linspace(xmin, xmax, num)
    #1 Graphe de la fonctions
    VY = f(VX)


    
    plt.plot(VX,VY,color='blue')

    # 2. Points et gradients sur l'axe
    liste_X, liste_grad = descente(f, grad_f, X0, delta=delta, nmax=nmax)
    for x in liste_X:    # points
        plt.scatter(x, 0, color='red')
    
    for i in range(len(liste_X)-1):    # flèches
        plt.arrow(liste_X[i],0, *(-delta*liste_grad[i]),0, linewidth=2, color='orange', length_includes_head=True, head_width=0.05, head_length=0.1)

    # 3. Points et gradients sur le graphe
    for x in liste_X:    # points
        plt.scatter(x, f(x), color='red')
    
    for i in range(len(liste_X)-1):    # flèches
        x = liste_X[i]
        xx = liste_X[i+1]
        plt.arrow(*x, *f(x), *(xx-x), *(f(xx)-f(x)), linewidth=2, color='orange', length_includes_head=True, head_width=0.05, head_length=0.1)

    # 4. affichage
    plt.scatter(0,0, color='blue')  # minimum
    # plt.colorbar();
    # plt.axis('equal')
    plt.tight_layout()
    # plt.savefig('descente_une_var_10.png')
    plt.show()
    return

