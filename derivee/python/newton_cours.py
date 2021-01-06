#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math

#------------------------------------------
# Newton
#------------------------------------------


def newton_iter(x0, func, derivee, n, cible=None):
    """
    Méthode de newton algorithme iteratif

    :param x0: terme initial
    :param func: function à etudier
    :param derivee: fonction derivee de func
    :param n: nombre d'iterations
    :param cible: la valeur que l'on recherche (sert uniquement a calculer l'erreur)
    :return: valuer qui approche la solution
    """
    x = x0
    for i in range(n):
        x = x - func(x)/derivee(x)
        if cible is not None:
            print(f"Etape {i} : {x} Erreur : {x - cible}")
    return x


def newton_rec(x0, func, derivee, n, cible=None):
    """
    Méthode de newton algorithme recursive

    :param x0: terme initial
    :param func: function à etudier
    :param derivee: fonction derivee de func
    :param n: nombre d'iterations
    :param cible: la valeur que l'on recherche (sert uniquement a calculer l'erreur)
    :return: valuer qui approche la solution
    """
    n = n - 1
    x = x0
    x = x - func(x)/derivee(x)
    if cible is not None:
        print(f"Etape {n} : {x} Erreur : {x - cible}")
    if n == 0:
        return x
    else:
        return newton_rec(x, func, derivee, n, cible=cible)


###########################
# Applications 
###########################
print("Calcul de racine cubique de 100 par la méthode de Newton")


# La fonction f(x)
def h(x):
    return x**3 - 100


# La fonction dérivée f'(x)
def h_prime(x):
    return 3*(x**2)


print("implementation iterative", newton_iter(10, h, h_prime, 9))
print("implementation recursive", newton_rec(10, h, h_prime, 9))


print("Calcul de 'l' où g atteint son minimum (là ou f=0) par la méthode de Newton")

# La fonction g(x)
def g(x):
    return x ** 2 - math.sin(x) + 1

# La fonction f(x) ( = g'(x) )
def f(x):
    return 2 * x - math.cos(x)

# La fonction dérivée f'(x)
def f_prime(x):
    return 2 + math.sin(x)

ell = 0.45018361129487355


ell = 0.45018361129487355
print("\nCalcul avec x0 = 0 et 10 iterations")
print("-- x0 = 0 --")
print(newton_iter(0, f, f_prime, 10, cible=ell))

print("-------------------------------")
print('l =', ell)
print('f(l) =', f(ell))
print('g(l) =', g(ell))

print("\nCalcul avec x0 = 10 et 10 iterations")
print("-- x0 = 10 --")
print(newton_rec(10, f, f_prime, 10, cible=ell))