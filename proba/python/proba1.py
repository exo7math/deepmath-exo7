
import numpy as np

X = np.array([3,1,8,6,0])


def softmax(X):
	Y = np.exp(X)
	return Y/np.sum(Y)


print("argmax :",np.argmax(X))

p = softmax(X)
for i in range(len(X)):
	print("p[{:d}] = {:.4f}".format(i,p[i]) )
