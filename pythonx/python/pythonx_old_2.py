
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------
# Les principales fonctions d'activations

def f1(x):
	return 1/(1+np.exp(-x))

def f2(x):
	return np.tanh(x)

def f3(x):
	return np.log(1+np.exp(x))


def elf4(x):
	if x >= 0:
		return 1
	else:
		return 0

f4 = np.vectorize(elf4, otypes=[np.float64])


def elf5(x):
	if x >= 0:
		return x
	else:
		return 0
f5 = np.vectorize(elf5, otypes=[np.float64])

def elf6(x):
	if x < 0:
		return np.exp(x)-1
	else:
		return x
f6 = np.vectorize(elf6, otypes=[np.float64])

a,b = -5,5
X = np.linspace(a,b,num=100)
Y = []
for f in [f1,f2,f3,f4,f5,f6]:
	Y = Y + [f(X)]

for i in range(6):
	plt.subplot(2,3,i+1)
	plt.plot(X,Y[i])


plt.savefig('activation.png')
plt.savefig('activation.pdf')
plt.show()



