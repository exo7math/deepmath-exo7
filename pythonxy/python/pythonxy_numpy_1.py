
import numpy as np

# 1. Tableaux à deux dimensions

# Définir des tableaux
A = np.array([[1, 2, 3], [4, 5, 6]])
print(A)
print(np.shape(A))

# Taille
# p = nb de lignes, n = nb colonnes
p,n = np.shape(A)   


# Parcourir tous les éléments
p, n = np.shape(A)
for i in range(p):
	for j in range(n):
		print(A[i,j])

# Fonctions
B = np.sqrt(A)
print(B)

# Définition automatique
Z = np.zeros((2,3))
print(Z)

# 2. Tableaux à deux dimensions (suite)

A = np.array([[1, 2, 3], [4, 5, 6]])

# Applatissement
X = A.flatten()
print(X)

# Remise en forme
AA = X.reshape((2,3))
print(AA)


B = X.reshape((3,2))
print(B)
