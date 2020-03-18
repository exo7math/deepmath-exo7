#!/usr/bin/python3

# Descente de gradient classique -- 2 variables
# Application à la régression linéaire


from descente import *


def exemple1():
    # points (0,3), (2,4), (6,6)
    # à trouver y = 1/2x+3, càd a= 1/2, b = 3
    points = [(0,3), (2,4), (6,6)]  # sur une droite
    # fonction erreur de 2 variables
    # somme des (y_i - (ax_i+b))^2
    def E(a, b):
        return sum([(y - (a*x+b))**2 for x,y in points])
    
    # gradient calculer par nos soins
    def grad_E(a, b):
        ga = sum([-2*x*(y - (a*x+b)) for x,y in points])
        gb = sum([-2*(y - (a*x+b)) for x,y in points])
        return np.array([ga, gb])

    # Test
    print("--- Descente de gradient ---")
    X0 = np.array([0, 1])
    mon_delta = 0.02
    affiche_descente(E, grad_E, X0, delta=mon_delta, nmax=11)
    graphique_descente_2var_2d(E, grad_E, X0, delta=mon_delta, zone = (-0.5,2.5,0.5,3.5))    
    graphique_regression(E, grad_E, X0, points, nmax=7, delta=mon_delta, zone = (-2,6,0,6)) 
    return


def exemple2():
    # à trouver environ y = 0.78 x - 2.46, càd a= 0.78, b = -2.46
    points = [(4,1), (7,3), (8,3), (10,6), (12,7)]  
    # fonction erreur de 2 variables
    # somme des (y_i - (ax_i+b))^2
    def E(a, b):
        return sum([(y - (a*x+b))**2 for x,y in points])
    
    # gradient calculer par nos soins
    def grad_E(a, b):
        ga = sum([-2*x*(y - (a*x+b)) for x,y in points])
        gb = sum([-2*(y - (a*x+b)) for x,y in points])
        return np.array([ga, gb])

    # Test
    print("--- Descente de gradient ---")
    X0 = np.array([3/4, -2])
    mon_delta = 0.001
    affiche_descente(E, grad_E, X0, delta=mon_delta, nmax=10000)
    # graphique_descente_2var_2d(E, grad_E, X0, delta=mon_delta, zone = (-0.5,1.5,-5,-2))    
    # graphique_regression(E, grad_E, X0, points, nmax=7, delta=mon_delta, zone = (0,15,0,6)) 
    return


def exemple_quadratique():
    points = [(0,2), (1,0), (2,0), (3,2)]  # sur une parabole
    # à trouver y = x^2 -3x + 2, càd  a = -3, b = 2
    # fonction erreur de 2 variables
    # somme des (y_i - (ax_i^2+bx_i+c))^2
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
    affiche_descente(E, grad_E, X0, delta=mon_delta ,nmax = 200)

    return


def exemple3():
    points = [(0,0,3), (1,0,4), (0,1,5)]  
    # à trouver z = ax + by + c, càd a= 1, b = 2, c = 3
    # fonction erreur de 3 variables
    # somme des (y_i - (ax_i+by_i+c))^2
    def E(a, b, c):
        return sum([(z - (a*x+b*y+c))**2 for x,y,z in points])
    
    # gradient calculer par nos soins
    def grad_E(a, b, c):
        ga = sum([-2*x*(z - (a*x+b*y+c)) for x,y,z in points])
        gb = sum([-2*y*(z - (a*x+b*y+c)) for x,y,z in points])
        gc = sum([-2*(z - (a*x+b*y+c)) for x,y,z in points])
        return np.array([ga, gb, gc])

    # Test
    print("--- Descente de gradient ---")
    X0 = np.array([0, 0, 0])
    mon_delta = 0.2
    affiche_descente(E, grad_E, X0, delta=mon_delta, nmax = 100)

    return


def exemple4():
    points = [(1,0,0), (0,1,5), (2,1,1), (1,2,0), (2,2,3)]  
    #  variation autour de z = ax + by + c, càd a= -1, b = 2, c = 2
    def E(a, b, c):
        return sum([(z - (a*x+b*y+c))**2 for x,y,z in points])
    
    # gradient calculer par nos soins
    def grad_E(a, b, c):
        ga = sum([-2*x*(z - (a*x+b*y+c)) for x,y,z in points])
        gb = sum([-2*y*(z - (a*x+b*y+c)) for x,y,z in points])
        gc = sum([-2*(z - (a*x+b*y+c)) for x,y,z in points])
        return np.array([ga, gb, gc])

    # Test
    print("--- Descente de gradient ---")
    X0 = np.array([0, 0, 0])
    mon_delta = 0.01
    affiche_descente(E, grad_E, X0, delta=mon_delta, nmax = 1000)

    return

# exemple1()
# exemple2()
exemple_quadratique()    
# exemple3()
# exemple4() 
