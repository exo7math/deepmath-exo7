
import numpy as np
import matplotlib.pyplot as plt

N = 100
f = np.sin(np.linspace(0,2*np.pi,N)) + 1*np.random.random(N)
g = 1/5*np.array([1,1,1,1,1])

h = np.convolve(f,g,'same')

ax = plt.subplot(2,1,1)
ax.set_title("Entrée f")
plt.plot(f)

ax = plt.subplot(2,1,2)
ax.set_title("Sortie h = f*g")
plt.plot(h,color='red')

# plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.95, hspace=1.0,wspace=0.5)
# plt.tight_layout()
# plt.savefig('pythonconv-1d.png')
plt.show()
