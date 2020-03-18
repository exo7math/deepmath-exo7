#!/usr/bin/python3

# Descente de gradient classique -- 2 variables
# Application à la régression linéaire


from descente import *
from descente_stochastique import *
from descente_lot import *

# Exemple pédagogique
points = [(2,0), (0,2), (4, 6), (1,0)]  # sur une parabole
# à trouver y = x^2 -3x + 2, càd  a = -3, b = 2

# Exemple ou sgd plus efficace
# N = 100
# vx = 5*np.random.random(N) - 1  # N valeurs x au hasard entre -1 et 4
# vy = vx**2 - 3*vx + 2       # les ordonnées y correspondant sur la paraboles
# points = [(vx[i],vy[i]) for i in range(N)]

# shuffle(points)


def exemple_1_classique(points):
    # à trouver y = x^2 -3x + 2, càd  a = -3, b = 2
    # fonction erreur de 2 variables
    # somme des (y_i - (x_i^2+ax_i+b))^2
    def E(a, b):
        return sum([(y - (x**2+a*x+b))**2 for x,y in points])
    
    # gradient calculer par nos soins 
    def grad_E(a, b):
        ga = sum([-2*x*(y - (x**2+a*x+b)) for x,y in points])
        gb = sum([-2*(y - (x**2+a*x+b)) for x,y in points])
        return np.array([ga, gb])

    # Test
    print("--- Descente de gradient ---")
    X0 = np.array([1, 1])
    mon_delta = 0.01
    affiche_descente(E, grad_E, X0, delta=mon_delta ,nmax = 200+1)
    graphique_descente_2var_2d(E, grad_E, X0, delta=mon_delta, zone = (-5,2,-2,4),nmax=2)    
    # graphique_regression(E, grad_E, X0, points, nmax=7, delta=mon_delta, zone = (-2,6,0,6)) 
    return


def exemple_1_stochastique(points):
    # à trouver y = x^2 -3x + 2, càd  a = -3, b = 2
    # fonction erreur de 2 variables
    # somme des (y_i - (x_i^2+ax_i+b))^2
    def E(a, b):
        return sum([(y - (x**2+a*x+b))**2 for x,y in points])
    
    # gradient calculer par nos soins 
    def grad_Ei(a, b, xi, yi):
        gia = -2*xi*(yi - (xi**2+a*xi+b))
        gib = -2*(yi - (xi**2+a*xi+b))
        return np.array([gia, gib])

    # Test
    print("\n --- Descente de gradient stochastique ---")
    X0 = np.array([1, 1])
    mon_delta = 0.01
    affiche_descente_stochastique(E, grad_Ei, points, X0, delta=mon_delta ,nmax = 4*200+1)
    graphique_descente_stochastique(E, grad_Ei, points, X0, delta=mon_delta, zone = (-5,2,-2,4),nmax=8)    
    # graphique_regression(E, grad_E, X0, points, nmax=7, delta=mon_delta, zone = (-2,6,0,6)) 
    return

def exemple_1_lot(points):
    # à trouver y = x^2 -3x + 2, càd  a = -3, b = 2
    # fonction erreur de 2 variables
    # somme des (y_i - (x_i^2+ax_i+b))^2
    def E(a, b):
        return sum([(y - (x**2+a*x+b))**2 for x,y in points])
    
    # gradient calculer par nos soins 
    def grad_Ei(a, b, xi, yi):
        gia = -2*xi*(yi - (xi**2+a*xi+b))
        gib = -2*(yi - (xi**2+a*xi+b))
        return np.array([gia, gib])

    # Test
    print("\n --- Descente de gradient par lots ---")
    X0 = np.array([1, 1])
    mon_delta = 0.01
    affiche_descente_lot(E, grad_Ei, points, 2, X0, delta=mon_delta , epoques=200)
    graphique_descente_lot(E, grad_Ei, points, 2, X0, delta=mon_delta, zone = (-5,2,-2,4), epoques=200)    
    # graphique_regression(E, grad_E, X0, points, nmax=7, delta=mon_delta, zone = (-2,6,0,6)) 
    return

# exemple_1_classique(points)
# exemple_1_stochastique(points)    
exemple_1_lot(points)    

