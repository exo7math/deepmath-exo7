
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


n = 100
VX = np.linspace(-1.0, 2.0, n)
VY = np.linspace(-1.0, 2.0, n)
X,Y = np.meshgrid(VX, VY)

def f(x,y):
    if (0 <= x <= 1) and (0 <= y <= 1):
	     return 1
    else:
        return 0

vec_f = np.vectorize(f)

Z = vec_f(X,Y)

fig = plt.figure()
ax = plt.axes(projection='3d')

# Fig 1. n = 5
ax.view_init(40, -30)
ax.plot_surface(X, Y, Z)
plt.tight_layout()
# plt.savefig('neurones-surface-1.png')

plt.show()