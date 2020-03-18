
import numpy as np

def max_pooling(A,k):
    n, p = A.shape
    B = A.reshape(n//k,k,p//k,k)
    C = B.transpose((0, 2, 1, 3))
    D = C.max(axis=(2,3))
    return D


def average_pooling(A,k):
    n, p = A.shape
    B = A.reshape(n//k,k,p//k,k)
    C = B.transpose((0, 2, 1, 3))
    D = C.mean(axis=(2,3))
    return D


# Un exemple 

A = np.arange(24)     # liste d'entier
np.random.shuffle(A)  # mélange aléatoire
A = A.reshape((4,6))  # matrice 

print("\n===== Exemple =====")
print("Matrice de départ :\n",A)
print("Taille de la matrice de départ", A.shape)

B = max_pooling(A,2)
print("Matrice après max-pooling :\n", B)
print("Taille de la matrice d'arrivée", B.shape)

C = average_pooling(A,2)
print("Matrice après pooling en moyenne :\n", C)
print("Taille de la matrice d'arrivée", C.shape)




################################
################################

# Exemple pas à pas pour comprendre comment
# fonctionne les fonctions ci-dessus 

def exemple():
    N = 6  # taille de la matrice initiale
    k = 3  # taille du pooling

    # matrice résultatnte sera de taille N//k

    A = np.arange(N**2).reshape((N,N))


    n, p = A.shape
    B = A.reshape(n//k,k,p//k,k)
    C = B.transpose((0, 2, 1, 3))
    D = C.max(axis=(2,3))


    print("===== Exemple pas à pas =====")
    print("Matrice de départ :\n", A)
    print("Taille de la matrice de départ", A.shape)

    print('Première sous matrice : \n', C[0,0])
    print('Max de cette sous matrice', C[0,0].max())
    print('Matrice des max : \n', C.max(axis=(2,3)))
    print('Matrice des moyennes :\n', C.mean(axis=(2,3)))

    return

exemple()