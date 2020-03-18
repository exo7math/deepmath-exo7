#!/usr/bin/python3

# Descente de gradient classique -- 1 variable

from descente import *



def exemple1():
    # fonction de 1 variable
    def f(x):
       return x**2+1
    
    # gradient = dérivée (mais sous la forme d'un vecteur de taille 1)
    def grad_f(x):
        return np.array([2*x])

    # Test
    print("--- Descente de gradient ---")
    X0 = np.array([2])
    # mon_delta = 0.2
    mon_delta = 0.9
    mon_delta = 1.1
    mon_delta = 0.05
    affiche_descente(f, grad_f, X0, delta=mon_delta, nmax=11)
    graphique_descente_1var(f, grad_f, X0, delta=mon_delta, nmax=11)    

    return


def exemple2():
    f = lambda x : x**4 -5*x**2 + x + 10  # sur [-2.0, 2.5]
    grad_f = lambda x :np.array([4*x**3 -10*x + 1])
    X0 = np.array([2]) 
    # X0 = np.array([0.5])
    # X0 = np.array([-0.5]) 
    # X0 = np.array([-2])          
    mon_delta = 0.02
    affiche_descente(f, grad_f, X0, delta=mon_delta, nmax = 10)
    graphique_descente_1var(f, grad_f, X0, delta=mon_delta, nmax = 5) 
    return


def exemple3():
    f = lambda x :  x**4 - 2*x**3 + 4  # sur [-1.5, 2.5]
    grad_f = lambda x :np.array([4*x**3 - 6*x**2])
    X0 = np.array([-1])
    X0 = np.array([-0.1])    
    mon_delta = 0.05
    affiche_descente(f, grad_f, X0, delta=mon_delta, nmax=1000)
    graphique_descente_1var(f, grad_f, X0, delta=mon_delta, nmax=100) 
    return   


exemple1()
# exemple2()    
# exemple3()  
