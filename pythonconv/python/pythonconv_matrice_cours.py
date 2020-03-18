##########################################
# Exemple du cours 

import numpy as np
from scipy import signal

A = np.array([[2,1,3,0],[1,1,0,5],[3,3,1,0],[2,0,0,2]])
M = np.array([[1,0,2],[2,1,0],[1,0,3]])

B = signal.convolve2d(A, M, mode='same', boundary='fill')

print("=== Exemple ===")
print('A =', A)
print('M =', M)
print('B =', B)



