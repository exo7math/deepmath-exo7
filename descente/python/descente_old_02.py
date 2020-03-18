#!/usr/bin/python3

# Descente de gradient classique -- 1 variable

import numpy as np
import matplotlib.pyplot as plt



# Exemples de fonctions avec leur 'gradient' (ici c'est la dérivée)
# f1 = x**2
# g1 = [2*x]
# delta = 0.9 donne zig-zag


# deux minimums
# f2 = x**4 -5*x**2 + x + 10  sur [-2.0, -2.5]
# g2 = 4*x**3 -10*x + 1

# point d'inflexion en 0
# f2 = x**4 - 2*x**3 + 1 [-2.0, -2.5]
# g2 = 4*x**3 - 6*x**2


# fonction de 1 variable
def f(x):
    fonc = x**4 - 2*x**3 + 1
    return fonc
    
# gradient calculer par nos soins
# pour raison de cohérence cela erste un vecteur (de longueur 1)
def gradient_f(x):
    grad = [4*x**3 - 6*x**2]
    return np.array(grad)


# descente inchangée
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
X0 = np.array([4])
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


def graphique_descente_1var(X0, delta=0.1, nmax=10):
    xmin, xmax = -1.0, 2
    num = 40
    VX = np.linspace(xmin, xmax, num)
    #1 Graphe de la fonctions
    VY = f(VX)
    plt.plot(VX,VY,color='blue')

    # 2. Points et gradients sur l'axe
    liste_X, liste_grad = descente(X0, delta=delta, nmax=nmax)
    for x in liste_X:    # points
        plt.scatter(x, 0, color='red')
    
    for i in range(len(liste_X)-1):    # flèches
        plt.arrow(liste_X[i],0, *(-delta*liste_grad[i]),0, linewidth=2, color='orange', length_includes_head=True, head_width=0.1, head_length=0.15)

    # 3. Points et gradients sur le graphe
    for x in liste_X:    # points
        plt.scatter(x, f(x), color='red')
    
    for i in range(len(liste_X)-1):    # flèches
        x = liste_X[i]
        xx = liste_X[i+1]
        plt.arrow(*x, *f(x), *(xx-x), *(f(xx)-f(x)), linewidth=2, color='orange', length_includes_head=True, head_width=0.1, head_length=0.15)

    # 4. affichage
    plt.scatter(0,0, color='blue')  # minimum
    # plt.colorbar();
    # plt.axis('equal')
    plt.tight_layout()
    # plt.savefig('descente.png')
    plt.show()
    return


# Test
X0 = np.array([0.1])
mon_delta = 0.05
affiche_descente(X0, delta=mon_delta)
graphique_descente_1var(X0, delta=mon_delta)