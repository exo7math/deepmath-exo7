# Produit tensoriel

import numpy as np



A = np.array([[1, 2], [3, 4]])
B = np.array([[2, -1], [1, 0]])
M = np.array([[1, 2], [0, 1]])

print("A = \n", A)
print("B = \n", B)
print("M = \n", M)

# Produit tensoriel par formule directe
MM = B @ M @ (A.T)

print("\nFormule directe")
print("MM = \n", MM)

# Produit tensoriel par produit de Kronecker
AB = np.kron(A, B)
vecM = M.flatten('F')  # vectorisation par colonnes
vecMM = AB @ vecM
MM = vecMM.reshape(2, 2).T  # remise en forme en matrice

print("\nProduit de Kronecker")
print("AB = \n", AB)
print("vecM = \n", vecM)
print("vecMM = \n", vecMM)
print("MM = \n", MM)




