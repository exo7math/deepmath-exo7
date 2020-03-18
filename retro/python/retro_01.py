#!/usr/bin/python3

# Descente de gradient classique -- 2 variables
# Application à la régression linéaire

from descente import *
from mpl_toolkits.mplot3d import Axes3D

# Fonction sigma
def sigma(x):
    return 1/(1+np.exp(-x))

# carré rouge
# rond bleus 
# à trouver environ y = 1/2x+1, càd a= 1/2, b = -1, c = 1  à un facteur multplicatif près
carres_rouges = [(1,1), (2,0.5), (2,2), (3,1.5), (3,2.75), (4,1), (4,2.5), (4.5,3), (5,1), (5,2.25)]
ronds_bleus   = [(0,3), (1,1.5), (1,4), (1.5,2.5), (2,2.5), (3,3.5), (3.5,3.25), (4,3), (4,4), (5,4)] 
N = len(carres_rouges) + len(ronds_bleus)    # taille des données
 
# fonction erreur de 3 variables
# somme des ((ax_i+by_i+c -0 ou -1))^2
def E(a, b, c):
    somme_rouges = sum([(sigma(a*x+b*y+c) - 1)**2 for x,y in carres_rouges])
    somme_bleus = sum([(sigma(a*x+b*y+c) - 0)**2 for x,y in ronds_bleus])
    return (somme_rouges + somme_bleus)/N

# gradient calculer par nos soins
def grad_E(a, b, c):
    ga_rouges = 1/N * sum([2*x*sigma(a*x+b*y+c)*(1-sigma(a*x+b*y+c))*(sigma(a*x+b*y+c)-1) for x,y in carres_rouges])
    gb_rouges = 1/N * sum([2*y*sigma(a*x+b*y+c)*(1-sigma(a*x+b*y+c))*(sigma(a*x+b*y+c)-1) for x,y in carres_rouges])
    gc_rouges = 1/N * sum([2*sigma(a*x+b*y+c)*(1-sigma(a*x+b*y+c))*(sigma(a*x+b*y+c)-1) for x,y in carres_rouges])

    ga_bleus = 1/N * sum([2*x*sigma(a*x+b*y+c)*(1-sigma(a*x+b*y+c))*(sigma(a*x+b*y+c)-0) for x,y in ronds_bleus])
    gb_bleus = 1/N * sum([2*y*sigma(a*x+b*y+c)*(1-sigma(a*x+b*y+c))*(sigma(a*x+b*y+c)-0) for x,y in ronds_bleus])       
    gc_bleus = 1/N * sum([2*sigma(a*x+b*y+c)*(1-sigma(a*x+b*y+c))*(sigma(a*x+b*y+c)-0) for x,y in ronds_bleus])        

    return np.array([ga_rouges+ga_bleus, gb_rouges+gb_bleus, gc_rouges+gc_bleus])

# Test
print("--- Descente de gradient ---")
X0 = np.array([0, 1, -2])
mon_delta = 1
affiche_descente(E, grad_E, X0, delta=mon_delta, nmax=11)

liste_X, liste_grad = descente(E, grad_E, X0, delta=mon_delta, nmax=10)
a, b, c = liste_X[-1]
print("Coefficients a, b, c du réseau et de la droite :", a, b, c)


# Affichage
def graphique_points():
    for x, y in carres_rouges:    # points
        plt.scatter(x, y, marker='s', color='red')
    for x, y in ronds_bleus:    # points
        plt.scatter(x, y, color='blue')

    VX = np.linspace(-0.5, 5.5, 100)
    a, b, c = 0, 1, -2  # init k = 0
    a, b, c =  0.077426, 0.8076023, -2.0058987  # k = 1

    a, b, c = 1.3288826, -1.8288267, 0.33387604  # k = 100
    a,b,c = 2.0328424, -3.8602521, 3.7039156  # final k = 1000
    VY = -1/b*(a*VX+c)

    #  affichage
    plt.plot(VX, VY, color='black')
    plt.axis('equal')
    plt.xlim(-0.5,5.5)
    plt.ylim(-0.5,5.5)
    plt.tight_layout()
    # plt.savefig('retro_01_e.png')
    plt.show()
    return

graphique_points()


def graphique_2d():
    a, b, c = 0, 1, -2  # init k = 0
    a, b, c = 1.3288826, -1.8288267, 0.33387604  # k = 100
    a,b,c = 2.0328424, -3.8602521, 3.7039156  # final k = 1000
    def f(x,y):
        return sigma(a*x+b*y+c)

    n = 50
    VX = np.linspace(-0.5, 5.5, n)
    VY = np.linspace(-0.5, 5.5, n)
    X,Y = np.meshgrid(VX, VY)
    Z = f(X,Y)


    # Contour - niveaux donnés avec valeurs

    mes_niveaux = np.linspace(0,1,20)
    plt.contourf(X, Y, Z,mes_niveaux, cmap='hot') 
    # plt.contourf(X, Y, Z,20,cmap='hot') 

    plt.colorbar();
    # trace = plt.contour(X, Y, Z, mes_niveaux)
    # plt.clabel(trace, inline=True, fontsize=8)

    for x, y in carres_rouges:    # points
        plt.scatter(x, y, marker='s', color='red')
    for x, y in ronds_bleus:    # points
        plt.scatter(x, y, color='blue')   

    plt.axis('equal')
    plt.xlim(-0.5,5.5)
    plt.ylim(-0.5,5.5)
    plt.tight_layout()
    # plt.savefig('retro_01_g.png')
    plt.show()

    return

graphique_2d()

# Affichage
def graphique_3d():
    a, b, c = 0, 1, -2  # init k = 0
    a, b, c = 1.3288826, -1.8288267, 0.33387604  # k = 100
    a,b,c = 2.0328424, -3.8602521, 3.7039156  # final k = 1000
    def f(x,y):
        return sigma(a*x+b*y+c)

    n = 50
    VX = np.linspace(-0.5, 5.5, n)
    VY = np.linspace(-0.5, 5.5, n)
    X,Y = np.meshgrid(VX, VY)
    Z = f(X,Y)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Fig 1. n = 5
    ax.view_init(30, -160)
    ax.plot_surface(X, Y, Z)


    # trace = plt.contour(X, Y, Z, mes_niveaux)
    # plt.clabel(trace, inline=True, fontsize=8)
    ax.set_xlabel("axe x")
    ax.set_ylabel("axe y")

    # plt.axis('equal')
    plt.xlim(-0.5,5.5)
    plt.ylim(-0.5,5.5)
    plt.tight_layout()
    # plt.savefig('retro_01_f.png')
    plt.show()
    return

graphique_3d()



a,b,c = 2.0328424, -3.8602521, 3.7039156  # final k = 1000
def F(x,y):
    return sigma(a*x+b*y+c)


print("F(0,3) =",F(0,3))
print("F(1,1) =",F(1,1))
print("F(4,3) =",F(4,3))
print("F(3,2.75) =",F(3,2.75))
print("F(2,3) =",F(2,3))
print("F(2,1) =",F(2,1))