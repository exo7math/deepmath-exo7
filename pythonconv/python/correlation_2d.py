
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

##########################################
# Correlation 2d

def retourner(M):
    M1 = M.flatten()
    M2 = M1[::-1]
    M3 = np.reshape(M2,np.shape(M))
    return M3

def convolution(f,g):
    return signal.convolve2d(f, g, mode='same')

def correlation(f,g):
    h = convolution(f,retourner(g))
    return h



##########################################
# Exemple

N = 4
A = np.arange(1,N**2+1).reshape((N,N))

N = 3
M = np.arange(1,N**2+1).reshape((N,N))

B = correlation(A,M)

print("=== Exemple ===")
print('A =', A)
print('M =', M)
print('B =', B)


##########################################
# Génération d'un signal émis par radar
# Cela peut être n'importe quoi

import imageio.v3 as imageio
icone = imageio.imread('smiley.png')

# Transformation en gris
epsilon = 0.2
icone = icone/128 -1  # valeurs entre -1 et 1
icone = epsilon*icone  # diminution des écarts (=> gris)
Noutx, Nouty = icone.shape
print('Taille du smiley ', icone.shape)

# Version image pour affichage
icone_image = 128 + 128*icone
plt.imshow(icone_image, cmap='gray', vmin=0, vmax=255)
plt.tight_layout()
# plt.savefig('correlation2d-1.png')
plt.show()


##########################################
# Génération d'un signal bruité reçus par le radar
# Cela peut être n'importe quoi


# Bruit
Nin = 500

# bruit = np.random.random((Nin,Nin))-0.5)

bruit = np.random.normal(0,1,(Nin,Nin))

# Version image pour affichage
bruit_image = 128 + 128*bruit
bruit_image = np.clip(bruit_image,0,255)
plt.imshow(bruit_image, cmap='gray', vmin=0, vmax=255, interpolation="none")
plt.tight_layout()
# plt.savefig('correlation2d-2.png')
plt.show()

# Ajout du smiley
posx, posy = 150, 250
iconebruit = bruit + np.pad(icone,((posy,Nin-Nouty-posy),(posx,Nin-Noutx-posx)))

# Version image pour affichage
iconebruit_image = 128 + 128*iconebruit
iconebruit_image = np.clip(iconebruit_image,0,255)
plt.imshow(iconebruit_image, cmap='gray', vmin=0, vmax=255, interpolation="none")
plt.tight_layout()
# plt.savefig('correlation2d-3.png')
plt.show()

# Correlation
correl = correlation(iconebruit,icone)

print("Min/max :",np.min(correl),np.max(correl))

plt.imshow(correl, cmap='gray', interpolation="none")
plt.tight_layout()
# plt.savefig('correlation2d-4.png')
plt.show()

