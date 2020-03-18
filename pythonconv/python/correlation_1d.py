
import numpy as np
import matplotlib.pyplot as plt

def correlation(f,g):
	gg = g[::-1]
	h = np.convolve(f,gg,'same')
	return h



##########################################
# Exemple de correlation

f = np.array([1,0,3,5,1])
g = np.array([1,2,3])  
h = correlation(f,g)

print("=== Exemple ===")
print('f =',f)
print('g =',g)
print('h =',h)

##########################################
# Génération d'un signal émis par radar
# Cela peut être n'importe quoi

Nout = 100
Xout = np.linspace(-3.14,3.14,Nout)
Yout = np.sinc(Xout) -0.5
plt.plot(Yout,color='red')
plt.tight_layout()
# plt.savefig('correlation1d-1.png')
plt.show()


##########################################
# Génération d'un signal bruité reçus par le radar
# Cela peut être n'importe quoi

a, b, Nin = 0, 1000, 1000
Xin = np.linspace(a, b, Nin)
# Yrandom = 1*np.sin(0.1*Xin) + 1*np.random.random(Nin)
Yrandom = 0.5*np.random.normal(0,1,Nin)
plt.figure(figsize=(10,3))
plt.plot(Yrandom)
plt.tight_layout()
# plt.savefig('correlation1d-2.png')
plt.show()


delay = 350


Yin = Yrandom + np.pad(Yout,(delay,Nin-Nout-delay))
plt.figure(figsize=(10,3))
plt.plot(Yin,color='blue')
plt.tight_layout()
# plt.savefig('correlation1d-3.png')
plt.show()


# Visualisation

def affichage_correlation(f,g):
	h = correlation(f,g)

	ax = plt.subplot(2,1,1)
	ax.set_title("Entrée f")
	plt.plot(f,color='blue')

	# ax = plt.subplot(3,1,2)
	# ax.set_title("fonction g")
	# plt.plot(g,color='orange')

	ax = plt.subplot(2,1,2)
	ax.set_title("Sortie h = correlation(f,g)")
	plt.plot(h,color='red')

	plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.95, hspace=1.0,wspace=0.5)
	plt.tight_layout()
	# plt.savefig('correlation1d-4.png')
	plt.show()
	return


affichage_correlation(Yin,Yout)
