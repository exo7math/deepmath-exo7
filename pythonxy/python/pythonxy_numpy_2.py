
import numpy as np

# 3. Fonctions deux variables

def f(x,y):
	return x**2 + y**2

n = 5
VX = np.linspace(0, 1, n)
VY = np.linspace(1, 1, n)
VZ = f(VX,VY)	
print(VZ)



# 4. Grille

n = 5
VX = np.linspace(0, 2, n)
VY = np.linspace(0, 2, n)
X,Y = np.meshgrid(VX, VY)

def f(x,y):
	return x**2 + y**2

Z = f(X,Y)

print(VX)
print(VY)
print(X)
print(Y)
print(Z)



# 5. Dessins (juste pour illustration pas code dans chapitre)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = plt.axes(projection='3d')
# ax.plot_surface(X, Y, Z)
ax.scatter(X, Y, Z, color = 'red')
ax.view_init(20, -150)
# plt.savefig('pythonxy-numpy1.png')
plt.show()