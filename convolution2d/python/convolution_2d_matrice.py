
import numpy as np
from scipy import signal


##########################################
# Exemple du cours (3d)

          
f = np.array([
	[0, 1, 1, 1, 0, 0, 0],
	[0, 0, 1, 1, 1, 0, 0],
	[0, 0, 0, 1, 1, 1, 0],
	[0, 0, 0, 1, 1, 0, 0],
	[0, 0, 1, 1, 0, 0, 0],
	[0, 1, 1, 0, 0, 0, 0],
	[1, 1, 0, 0, 0, 0, 0]])
g = np.array([[1,0,2],[0,1,0],[0,1,1]])

h = signal.convolve2d(f, g, boundary='fill', mode='same')

print("=== Exemple ===")
print('f =',f)
print('g =',g)
print('h =',h)


##########################################
# Exemple du cours : 
f = np.array([[2,1,3,0],[1,1,0,5],[3,3,1,0],[2,0,0,2]])
g = np.array([[1,0,2],[2,1,0],[1,0,3]])

h = signal.convolve2d(f, g, boundary='fill', mode='same')

print("=== Exemple ===")
print('f =',f)
print('g =',g)
print('h =',h)


##########################################
# Exemple du cours
f = np.array([[2,1,3,0],[1,1,0,5],[3,3,1,0],[2,0,0,2]])
g = np.array([[0,0,0],[0,1,2],[0,3,4]])
g = np.array([[1,2],[3,4]])

h = signal.convolve2d(f, g, boundary='fill', mode='same')

print("=== Exemple ===")
print('f =',f)
print('g =',g)
print('h =',h)


##########################################
# Exemple du cours
f = np.array([[2,-1,7,3,0],[2,0,0,-2,1],[-5,0,-1,-1,4]])
g = np.array([[-2,-1,0],[-1,1,1],[0,1,2]])  #  'emboss'

h = signal.convolve2d(f, g, boundary='fill', mode='same')

print("=== Exemple ===")
print('f =',f)
print('g =',g)
print('h =',h)

##########################################
# Exemple du cours : translation

N = 4
f = np.arange(1,N**2+1).reshape((N,N))

# g = np.array([[0,0,0],[0,1,0],[0,0,0]])  # identité
g = np.array([[0,0,0],[0,0,0],[0,1,0]])    # vers le haut
# g = np.array([[0,0,0],[0,0,1],[0,0,0]])  # vers la droite

h = signal.convolve2d(f, g, boundary='fill', mode='same')


print("=== Exemple ===")
print('f =',f)
print('g =',g)
print('h =',h)

##########################################
# Exemple du cours : moyenne

N = 5
f = np.arange(1,N**2+1).reshape((N,N))

g = 1/9*np.ones((3,3))    # vers le haut

h = signal.convolve2d(f, g, boundary='fill', mode='same')


print("=== Exemple ===")
print('f =',f)
print('g =',g)
print('h =',h)