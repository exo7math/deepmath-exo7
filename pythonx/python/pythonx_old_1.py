
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------
# 2. Plusieurs dessins

a,b = 0,1
X = np.linspace(a,b,num=100)
Y1 = X
Y2 = X**2
Y3 = X**3


plt.subplot(3,1,1)
plt.title('x^1')
plt.plot(X,Y1)

plt.subplot(3,1,2)
plt.title('x^2')
plt.plot(X,Y2)

plt.subplot(3,1,3)
plt.title('x^3')
plt.plot(X,Y3)

plt.show()

