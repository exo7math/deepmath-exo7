
import numpy as np
import matplotlib.pyplot as plt

##########################################
# Exemple du cours

f = np.array([1,0,3,5,1])
g = np.array([1,2,3])  
h = np.convolve(f,g,'same')

print("=== Exemple ===")
print('f =',f)
print('g =',g)
print('h =',h)

##########################################
# Exemple du cours

f = np.array([1,0,3,5,1])
g = np.array([1,2,3])  
h = np.convolve(f,g,'full')

print("=== Exemple ===")
print('f =',f)
print('g =',g)
print('h =',h)




# Visualisation

def affichage_convolution(f,g,mode='same'):
	h = np.convolve(f,g,mode=mode)

	ax = plt.subplot(2,1,1)
	ax.set_title("Entrée f")
	plt.plot(f)

	# ax = plt.subplot(3,1,2)
	# ax.set_title("fonction g")
	# plt.plot(g,color='orange')

	ax = plt.subplot(2,1,2)
	ax.set_title("Sortie h = f*g")
	plt.plot(h,color='red')

	plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.95, hspace=1.0,wspace=0.5)
	# plt.tight_layout()
	# plt.savefig('convolution-1d-2.png')
	plt.show()
	return

# Exemple 1

f = np.array([4,2,1,4,5,1,3])
g = 1/3*np.array([1,1,1])  
# affichage_convolution(f,g,mode='same')

# Exemple 2 : moyenne mobile et bruit
N = 100
f = np.sin(np.linspace(0,2*np.pi,N)) + 1*np.random.random(N)
g = 1/5*np.ones(5)
affichage_convolution(f,g,mode='same')

