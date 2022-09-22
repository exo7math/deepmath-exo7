
import numpy as np
import matplotlib.pyplot as plt

# Partie A - Importer une image comme un tableau
import imageio.v3 as imageio
A = imageio.imread('image_avant.png')

# Partie B - Motif de convolution
M = np.array([[-2,-1,0],[-1,1,1],[0,1,2]])

# Partie C - Calcul de la convolution
from scipy import signal
B = signal.convolve2d(A, M, boundary='fill', mode='same')

# Partie D - Affichage des images avant/après
fig = plt.figure(figsize = (10,5))

ax = plt.subplot(1,2,1)
ax.set_title("Image originale")
ax.imshow(A, cmap='gray')

ax = plt.subplot(1,2,2)
ax.set_title("Image sortie")
plt.imshow(B, cmap='gray')

plt.show()

# Partie E - Sauvegarde de l'image
B = np.clip(B,0,255)    # limite les valeurs entre 0 et 255
B = B.astype(np.uint8)  # conversion en entiers
imageio.imwrite('image_apres.png', B)

