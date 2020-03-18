
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n = 50
VX = np.linspace(-3.14, 3.14, n)
VY = np.linspace(-3.14, 3.14, n)
X,Y = np.meshgrid(VX, VY)

def f(x,y):  
	z = np.cos(x)*np.sin(y)+2
	return z

Z = f(X,Y)


# Contours et surface
fig = plt.figure()
ax = plt.axes(projection='3d')
plt.axis('off')
# ax.grid(False)
ax.view_init(40, -115)

mes_niveaux = np.linspace(1,3,10)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none',alpha=1)
x, y, z = 1,0, f(1,0)
# ax.arrow((x,y,z), (-1,-1), linewidth=2, color='orange', length_includes_head=True, head_width=0.05, head_length=0.1)
# ax.quiver(x, y, z+1, -1, -1, -0.5,colors='black', linewidth=2,linestyles="solid",arrow_length_ratio=0.1)
plt.tight_layout()
# plt.savefig('descente_intro.png')
plt.show()

