# Plongement positionnel

import numpy as np
import matplotlib.pyplot as plt


n = 10  # dimension du plongement
K = 4   # nombre de mots à plonger
p = 100 # nombre de divisions



### Partie A : Plongement positionnel

def period_omega(j, n, p=100):
    return 1/np.power(p, 2*j/n)


def position_matrix(K, n, p=100):
    P = np.zeros((K, n))
    for k in range(K):
        for j in range(n//2):
            omega = period_omega(j, n, p=p)
            P[k, 2*j] = np.cos(k*omega)
            P[k, 2*j+1] = np.sin(k*omega)

    return P

# Test
def exemple_position():
    P = position_matrix(K=4, n=6, p=100)
    print(P.shape)
    print("n =", n, "K =", K)
    print("Matrice des plongements positionnels (transposée) :")
    print(P.T)
    return

# exemple_position()

def exemple_position_bis():
    P = position_matrix(K=4, n=6, p=100)
    print("n =", n, "K =", K)
    for k in range(K):
        print(f"Plongement de la position {k} :")
        print(P[k, :])
    return

# exemple_position_bis()



### Partie B : Position relative et rotation

def rotation_matrix(omega):
    """ Matrice de rotation d'angle omega """
    return np.array([[np.cos(omega), -np.sin(omega)],
                     [np.sin(omega), np.cos(omega)]])


def blocks_rotation_matrix(l, K, n, p=100):
    """ Matrice de rotation ayant des blocs de taille 2 correspondant à un décalage de longueur l """
    R = np.zeros((n, n))
    for j in range(n//2):
        omega = period_omega(j, n, p=p)
        R[2*j:2*j+2, 2*j:2*j+2] = rotation_matrix(l*omega)
    return R


# Test
def exemple_rotation():
    P = position_matrix(K=4, n=4, p=100)
    l = 2
    R = blocks_rotation_matrix(l=l, K=4, n=4, p=100)
    # print(R.shape)
    print("Matrice de rotation par blocs :")
    print(R)

    k = 0
    Pk = P[k, :]
    Pl = P[k+l, :]
    Pll = R @ Pk
    print("Vecteur position au rang k + l")
    print("  Valeur directe : ", Pl)
    print("  Par rotation   : ", Pll)

    return

# exemple_rotation()


### Partie C : Visualisation

def plot_position_matrix(P):
    """ Visualisation de la matrice de position """
    plt.figure(figsize=(10, 10))
    plt.imshow(P, cmap='hot', interpolation='nearest')

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig("position-matrix.png", dpi=600)
    plt.show()
    return

P = position_matrix(K=100, n=100, p=10000)
plot_position_matrix(P.T)


def plot_kcst():
    """ Visualisation des fonctions avec k constant """

    p = 10000
    n = 100

    plt.figure(figsize=(10, 10))
    X = np.linspace(0, 20, 200)
    # Affichage sur des sous-graphes
    lmax = 5
    for l in range(0, lmax):
        plt.subplot(lmax, 1, l+1)
        k = 10*l
        Omega = 1/np.power(p, 2*X/n)
        plt.plot(X, np.sin(k*Omega))
        plt.xticks([])
        plt.yticks([])
        plt.title(f"k = {k}")

    plt.tight_layout()
    # plt.savefig("position-kcst.png", dpi=600)
    plt.show()
    return

# plot_kcst()


def plot_icst():
    """ Visualisation des fonctions avec i constant """

    p = 10000
    n = 100

    plt.figure(figsize=(10, 10))
    X = np.linspace(0, 50, 200)
    # Affichage sur des sous-graphes
    imax = 5
    for i in range(0, imax):
        plt.subplot(imax, 1, i+1)
         
        omega = 1/np.power(p, 2*i/n)
        plt.plot(X, np.cos(X*omega))

        plt.xticks([])
        plt.yticks([])    
        plt.title(f"i = {i}")

    plt.tight_layout()
    # plt.savefig("position-icst.png", dpi=600)
    plt.show()
    return

# plot_icst()


def plot_surface():
    """ Visualisation de la surface (x,y) -> sin(x/y) """
    cX =  np.linspace(0.05, 3, 100)
    cY =  np.linspace(0.05, 3, 100)
    X, Y = np.meshgrid(cX, cY)
    Z = np.sin(X/Y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  
    ax.plot_surface(X, Y, Z, cmap='viridis')
    plt.show()
    # plt.savefig("position-surface.png", dpi=600)
    return

# plot_surface()




### Annexe : Motivation

def digits(k, b):
    """ Retourne les chiffres de n en base b """
    liste_d = []
    i = 0
    while True:
        d = (k // b**i) % b
        if b**i > k:
            break
        i += 1
        liste_d.append(d)
    liste_d.reverse()
    return liste_d

# test
def exemple_digits():
    k = 23
    b = 2
    print("A la main :", digits(k, b))
    print("Python :", bin(k))
    return

# exemple_digits()
