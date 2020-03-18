
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------
# 4. Plusieurs fonctions sur le même dessin

def f(x,k):
	return 1/(1+np.exp(-k*x))

a,b = -5,5
X = np.linspace(a,b,num=100)

for k in range(1,5):
	Y = f(X,k)
	plt.plot(X,Y, label="k={}".format(k))

plt.legend()
plt.savefig('pythonx-sigma.png')
plt.show()



