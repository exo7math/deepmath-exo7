
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------
# 5. Fonctions mathématiques de numpy

print("\n\n--- 1. Fonctions mathématiques de numpy ---\n")

X = np.array([0,1,2,3,4,5])

print(X**2)
print(np.sqrt(X))

print(np.exp(X))
print(np.cos(X))  # en radians
print(np.cos(2*np.pi/360*X))  # en degrés

# Ne pas utiliser le mode math
# import math  
# print(math.cos(X))   # ERREUR
