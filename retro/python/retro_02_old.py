#!/usr/bin/python3

# Préparation pour retro_02_tf.py

from descente import *

from tensorflow.keras import backend as K

# Fonction tanh
# sa dérivée tanh'(x) = 1 - tanh(x)**2
def tanh(x):
    E = np.exp(-2*x)
    return (1-E)/(1+E)

# Fonction sigma
def sigma(x):
    return 1/(1+np.exp(-x))

# Fonction heaviside
# définie dans 'descente'
def heaviside(x):
    return np.heaviside(x,1)

# carré rouge +1
# rond bleus -1
# à trouver : fonction XOR : +1 pour (x>0, y>0) et pour (x<0, y<0) ; -1 sur (x>0, y<0) et (x<0, y>0)
carres_rouges = [(0.0,0.4), (0.1,0.7), (0.2,0.2), (0.2,1.0), (0.4,0.6), (0.6,0.0), (0.6,0.4), (0.8,0.9), (1.0,0.1), (1.0,0.5)]
ronds_bleus   = [(0.1,-0.3), (0.1,-0.7), (0.2,-0.1), (0.3,-0.8), (0.6,-0.1), (0.7,-0.3), (0.8,-0.2), (0.8,-1.0), (0.9,-0.9), (1.0,-0.1) ] 
carres_rouges += [(-x,y) for x,y in ronds_bleus]  # symétriques
ronds_bleus  += [(-x,y) for x,y in carres_rouges]  # symétriques

# Fonction produite par le neurone
def f(x,y,poids,phi):
    a1,b1,c1,a2,b2,c2,a3,b3,c3 = poids
    S1 = phi(a1*x+b1*y+c1) # sortie de premier neurone
    S2 = phi(a2*x+b2*y+c2) # sortie du second neurone
    val  = phi(a3*S1+b3*S2+c3)
    return val


# Affichage
def graphique_points():


    poids_xor = (1,1,-0.5,-1,-1,1.5,1,1,-1.5)  # NON ca c'est pour Heaviside et 0/1

    phi = tanh
    poids_tf =  [-10.856818, -12.307046, 12.102052, 13.08996, 18.090952, 14.482832, -5.426733, -4.785306, 9.468286]  
    
    # phi = sigma
    # poids_tf = [-11.091958, -12.72977, 12.387876, 12.768436, 17.153465, 13.889489, -9.536562, -9.757105, 17.659502]

    phi = heaviside
    poids_heavisde = (1,1,0.5, -1,-1,1.5, 1,1,-1.5)

    n = 50
    VX = np.linspace(-1.0, 1.0, n)
    VY = np.linspace(-1.0, 1.0, n)
    X,Y = np.meshgrid(VX, VY)
    Z = f(X,Y,poids_tf,phi)


# Contour - niveaux donnés avec valeurs

    mes_niveaux = np.linspace(-1,1,20)
    plt.contourf(X, Y, Z,mes_niveaux, cmap='hot') 
    plt.colorbar();
    # trace = plt.contour(X, Y, Z, mes_niveaux)
    # plt.clabel(trace, inline=True, fontsize=8)

    for x, y in carres_rouges:    # points
        plt.scatter(x, y, marker='s', color='red')
    for x, y in ronds_bleus:    # points
        plt.scatter(x, y, color='blue')   



    # plt.axis('equal')
    plt.xlim(-1.2,1.2)
    plt.ylim(-1.2,1.2)
    plt.tight_layout()
    # plt.savefig('retro_02.png')
    plt.show()
    return

graphique_points()
