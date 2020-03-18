
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------
# 1. Un exemple

def f(x):
	return np.exp(-x**2)

a,b = -3,3
X = np.linspace(a,b,num=100)
Y = f(X)

plt.plot(X,Y)
# plt.savefig('pythonx-gauss.png')
plt.show()
