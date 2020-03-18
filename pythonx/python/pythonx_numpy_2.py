
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------
# 1. Définition d'un vecteur (tableau à une dimension)

print("\n\n--- 1. Définition d'un vecteur ---\n")
# Par 'array'
X = np.array([1,2,3,4])
print(X)

# Ce n'est pas une liste python
print(type(X))
print(type([1,2,3,4]))
print(X==[1,2,3,4])

# Par 'arange'
X = np.arange(0,100,5)
print(X)
X = np.arange(1,8,0.5)
print(X)

# Par 'linspace'
X = np.linspace(0,1,num=12)
print(X)
print(len(X)) 

# ----------------------------------------------------
# 2. Opérations élémentaires
print("\n\n--- 2. Opérations élémentaires ---\n")

X = np.array([1,2,3,4])
print(2*X)
print(X+1)
print(X**2)
print(1/X)
print(np.sum(X))

Y = np.array([10,11,12,13])
print(X+Y)
print(X*Y)

print(np.max(X))
print(np.min(X))
print(np.sum(X))

# ----------------------------------------------------
# 3. Définition d'un vecteur (suite)
print("\n\n--- 3. Définition d'un vecteur (suite) ---\n")

# Par 'zeros'
X = np.zeros(5)
print(X)

# Par 'ones'
X = np.ones(5)
print(X)
print(7*X)

# Par 'random'
X = np.random.random(5)
print(X)


# ----------------------------------------------------
# 4. Utilisation comme une liste
print("\n\n--- 4. Utilisation comme une liste ---\n")
X = np.linspace(1,2,num=10)
print(X)

# Récupérer un élément
print(X[1])

# Parcourir tous les éléments :
for x in X:
	print(x)

# Longueur
print(len(X))
print(np.shape(X))

for i in range(len(X)):
	print(i,X[i])

