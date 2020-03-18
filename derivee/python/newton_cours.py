#!/usr/bin/python3
# -*- coding: utf-8 -*-

#------------------------------------------
# Newton
#------------------------------------------

# La fonction f(x)
def f(x):
    return x**3 - 100

# La fonction dérivée f'(x)
def df(x):
    return 3*(x**2)    


# Méthode de Newton 
# x0 est le terme initial, n est le nombre d'étapes
def newton(x0,n):  
    x = x0
    for i in range(1,n+1):
        x = x - f(x)/df(x)
        # print("Etape ", i, " : ",x,"Erreur :",x-100**(1/3)) 
        # print("Etape ", i, " : ",x,"Erreur :",x-100**(1/3))
    return x


    
print("Calcul de racine cubique de 100 par la méthode de Newton")
print(newton(10,9))  

#------------------------------------------
# Newton
#------------------------------------------


# Calcul de racine cubique de 100 avec 1000 décimales

# Module décimale pour avoir une grande précision 
from decimal import *

# Precision souhaitée (par exemple 1010 décimales pour éviter les erreurs d'arrondi)
getcontext().prec = 1010


# Cas de la racine carrée sqrt(a)
# n est le nombre d'itérations
def newton_bis(x0,n):  
    x = Decimal(x0)
    for i in range(1,n+1):
        x = x - f(x)/df(x)
        # print("Etape ", i, " : ", x) 
        # print("Etape ", i, " : ",x,"Erreur :",x**3-100)
    return x


# Exemple 
n=13
sqrt3_100 = newton_bis(10,n)
# print("Racine cubique de 100 (n=%d) : " %n, sqrt3_100)

# Test
# print(sqrt3_100-Decimal(100**1/3))
# print(sqrt3_100)
print(sqrt3_100**3)



#------------------------------------------
# Newton
#------------------------------------------

from math import *

# La fonction g(x)
def g(x):
    return x**2 - sin(x) + 1 

# La fonction f(x) ( = g'(x) )
def f(x):
    return 2*x - cos(x) 

# La fonction dérivée f'(x)
def df(x):
    return 2 + sin(x)

ell = 0.45018361129487355

def newton(x0,n):  
    x = x0
    for i in range(1,n+1):
        x = x - f(x)/df(x)
        print("Etape ", i, " : ",x,"Erreur :",x-ell)
    return x

print("Calcul de 'l' où g atteint son minimum (là ou f=0) par la méthode de Newton")

print("-- x0 = 0 --")
ell = newton(0,10)
ell = 0.45018361129487355
print('l =',ell)
print('f(l) =',f(ell)) 
print('g(l) =',g(ell)) 

print("-- x0 = 10 --")
ell = newton(10,10)