
import numpy as np
from scipy import signal



##########################################
# Exemple du cours : convolution même taille

A = np.array([[2,1,3,0],[1,1,0,5],[3,3,1,0],[2,0,0,2]])
M = np.array([[1,0,2],[2,1,0],[1,0,3]])

B = signal.convolve2d(A, M, mode='same', boundary='fill')

print("=== Exemple : convolution même taille ===")
print('A =',A)
print('M =',M)
print('B =',B)



##########################################
# Exemple du cours : convolution étendue

A = np.array([[2,1,3,0],[1,1,0,5],[3,3,1,0],[2,0,0,2]])
M = np.array([[1,0,2],[2,1,0],[1,0,3]])

B = signal.convolve2d(A, M, mode='full')

print("=== Exemple : convolution étendue ===")
print('A =',A)
print('M =',M)
print('B =',B)



##########################################
# Exemple du cours : convolution restreinte au domaine de validité

A = np.array([[2,1,3,0],[1,1,0,5],[3,3,1,0],[2,0,0,2]])
M = np.array([[1,0,2],[2,1,0],[1,0,3]])

B = signal.convolve2d(A, M, mode='valid')

print("=== Exemple : convolution restreinte ===")
print('A =',A)
print('M =',M)
print('B =',B)


##########################################
# Exemple
          
A = np.array([
	[0, 1, 1, 1, 0, 0, 0],
	[0, 0, 1, 1, 1, 0, 0],
	[0, 0, 0, 1, 1, 1, 0],
	[0, 0, 0, 1, 1, 0, 0],
	[0, 0, 1, 1, 0, 0, 0],
	[0, 1, 1, 0, 0, 0, 0],
	[1, 1, 0, 0, 0, 0, 0]])


M = np.array([[1,0,2],[0,1,0],[0,1,1]])

B = signal.convolve2d(A, M, boundary='fill', mode='same')

print("=== Exemple ===")
print('A =',A)
print('M =',M)
print('B =',B)


##########################################
# Exemple
A = np.array([[2,1,3,0],[1,1,0,5],[3,3,1,0],[2,0,0,2]])
M = np.array([[0,0,0],[0,1,2],[0,3,4]])
M = np.array([[1,2],[3,4]])

B = signal.convolve2d(A, M, boundary='fill', mode='same')

print("=== Exemple ===")
print('A =',A)
print('M =',M)
print('B =',B)


##########################################
# Exemple
A = np.array([[2,-1,7,3,0],[2,0,0,-2,1],[-5,0,-1,-1,4]])
M = np.array([[-2,-1,0],[-1,1,1],[0,1,2]])  #  'emboss'

B = signal.convolve2d(A, M, boundary='fill', mode='same')

print("=== Exemple ===")
print('A =',A)
print('M =',M)
print('B =',B)


##########################################
# Exemple du cours : translation

N = 4
A = np.arange(1,N**2+1).reshape((N,N))

# M = np.array([[0,0,0],[0,1,0],[0,0,0]])  # identité
M = np.array([[0,0,0],[0,0,0],[0,1,0]])    # vers le haut
# M = np.array([[0,0,0],[0,0,1],[0,0,0]])  # vers la droite

B = signal.convolve2d(A, M, boundary='fill', mode='same')


print("=== Exemple ===")
print('A =',A)
print('M =',M)
print('B =',B)


##########################################
# Exemple du cours : moyenne

N = 5
A = np.arange(1,N**2+1).reshape((N,N))

M = 1/9*np.ones((3,3))    # vers le haut

B = signal.convolve2d(A, M, boundary='fill', mode='same')


print("=== Exemple ===")
print('A =',A)
print('M =',M)
print('B =',B)