
import numpy as np
from scipy import signal



##########################################
# Convolution étendue
def convolution(A, M):
    return signal.convolve2d(A, M, mode='full')   


##########################################
# Matrices aléatoires de taille n x p
n, p = 6, 4
A = np.random.randint(0, 10, size = (n,p))


# Motifs aléatoires de taille 3 x 3
M = np.random.randint(-5,5, size = (3,3))
N = np.random.randint(-5,5, size = (3,3))


##########################################
# Exemple d'associativité

# Calcul de (A*M)*N
AM = convolution(A, M)
AMN1 = convolution(AM, N)

# Calcul de A*(M*N)
MN = convolution(M, N)
AMN2 = convolution(A, MN)

print('A =', A)
print('M =', M)
print('N =', N)
print('(A*M)*N =', AMN1)
print('A*(M*N) =', AMN2)

print("Associativité pour la convolution ?\n", AMN1==AMN2)





##########################################
# Retournement

def retourner(M):
    M1 = M.flatten()
    M2 = M1[::-1]
    M3 = np.reshape(M2,np.shape(M))
    return M3

##########################################
# Retournement

def correlation(A, M):
    B = convolution(A,retourner(M))
    return B

##########################################
# Exemple de non associativité pour la correlation

n, p = 4, 4
A = np.random.randint(0, 10, size = (n,p))
M = np.random.randint(-5,5, size = (3,3))
N = np.random.randint(-5,5, size = (3,3))


##########################################
# Exemple d'associativité

# Calcul de (A*M)*N
AM = correlation(A, M)
AMN1 = correlation(AM, N)

# Calcul de A*(M*N)
MN = correlation(M, N)
AMN2 = correlation(A, MN)

print('A =', A)
print('M =', M)
print('N =', N)
print('(A*M)*N =', AMN1)
print('A*(M*N) =', AMN2)

print("Associativité pour la corrélation ?\n", AMN1==AMN2)