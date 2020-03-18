
# Tenseur avec numpy

import numpy as np

# Partie A - Qu'est ce qu'un tenseur ?

print("=== Partie A - Qu'est ce qu'un tenseur ? ===")

# Vecteur : 1-tenseur
V = np.array([5, 7, 8])
print(V)
print("Dimension (ou rang) :", V.ndim)
print("Taille :", V.shape)
print("Nb d'éléments :", V.size)
print(V[0])  # premier élément


# Matrice : 2-tenseur
A = np.array([[6, 0, 2], 
	          [3, 2, 4]])
print(A)
print("Dimension (ou rang) :", V.ndim)
print("Taille :", A.shape)
print("Nb d'éléments :", A.size)
print(A[0])
print(A[0,0])  # element à la place (0,0)


# 3-tenseur
T = np.array([ [[5, 1, 0], 
	            [3, 3, 2]],
	           [[1, 1, 0], 
	            [5, 1, 7]], 
	           [[0, 0, 1], 
	            [8, 1, 9]] ])

print(T)
print("Dimension (ou rang) :", T.ndim)
print("Taille :", T.shape)
print("Nb d'éléments :", T.size)
print(T[2])
print(T[2,1])
print(T[2,1,2])


# 4-tenseur
uns = np.ones((3,3,3,3))
img = np.random.randint(0,5,(10,32,32,3))


# Partie B - Opération sur les tenseurs

print("=== Partie B - Opération sur les tenseurs ===")

# Opérations sur les éléments
VV = V + 3
AA = A ** 2
TT = T - 1
print(VV)
print(AA)
print(TT)

# Opérations élément par élément entre tenseurs de même taille
print(V+VV)
print(A*AA)
print(T+TT)

# Fonctions sur les tenseurs 

print(V.mean())
print(A.sum())
print(np.sqrt(T))


# Partie C - Changer de taille
print("=== Partie C - Changer de taille ===")

# Mise sous forme de vecteur
V1 = A.flatten()
V2 = T.flatten()
print(V1.shape)
print(V2.shape) 
print(V2)

# Reformatage
print(T.size)  # 18
print(T.shape) # (3,2,3)
T1 = np.reshape(T,(2,9))
T2 = np.reshape(T,(2,3,3))
# T4 = np.reshape(T,(9,2))
# T5 = np.reshape(T,(3,3,2))

# Laisser numpy calculer la longuer d'un axe tout seul avec '-1'
T3 = np.reshape(T,(9,-1))
print(T3.shape)
print('----')
print('T1',T1)
print('T2',T2)
print('T3',T3)
print('----')
# Mettre à la même forme
V = np.array([1, 2, 3, 4])
A = np.array([[1, 0,], 
	          [-2, 5]])
B = np.reshape(V,A.shape)


# Regrouper deux tenseurs
A = np.arange(1,7).reshape((2,3))
B = np.arange(7,13).reshape((2,3))
print(A)
print(B)
C1 = np.concatenate((A, B), axis=0)
C2 = np.concatenate((A, B), axis=1)
print(C1) 
print(C2)
