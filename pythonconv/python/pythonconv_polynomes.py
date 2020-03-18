
import numpy as np
import matplotlib.pyplot as plt

##########################################
# Exemple du cours

P = np.array([1,2,3,4])
Q = np.array([5,6,7])  
R = np.convolve(P,Q,'full')
RR = np.polymul(P,Q)

print("=== Exemple ===")
print('P =', P)
print('Q =', Q)
print('R =', R)
print('RR =', RR)
