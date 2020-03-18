#!/usr/bin/python3

# Descente de gradient classique -- 2 variables

from descente import *
from descente_stochastique import *
from descente_lot import *





def exemple1():
    # fonction de 2 variables
    def f(x, y):
        return x**2 + 3*y**2
    
    # gradient calculer par nos soins
    def grad_f(x, y):
        g = [2*x, 6*y]
        return np.array(g)

    # Test
    print("--- Descente de gradient ---")
    X0 = np.array([2, 1])    
    mon_delta = 0.2
    X0 = np.array([-1, -1])    
    mon_delta = 0.1    
    affiche_descente(f, grad_f, X0, delta=mon_delta, nmax = 21)
    graphique_descente_2var_2d(f, grad_f, X0, delta=mon_delta, nmax = 10, zone = (-2.5,2.5,-1.5,1.5) ) 

    return


def exemple2():
    f = lambda x, y : x**2 + (x-y**2)**2
    grad_f = lambda x, y :np.array([2*x+2*(x-y**2), -4*y*(x-y**2)])
    X0 = np.array([2.2, 0.6])
    mon_delta = 0.2
    affiche_descente(f, grad_f, X0, delta=mon_delta, nmax=10)
    graphique_descente_2var_2d(f, grad_f, X0, delta=mon_delta, nmax = 0, zone = (-2,3,-2,2)) 
    return


# exemple1()
exemple2()    

