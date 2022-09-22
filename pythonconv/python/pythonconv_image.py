
import numpy as np
import matplotlib.pyplot as plt


# Partie A - Importer une image comme un tableau
import imageio.v3 as imageio
f = imageio.imread('image_avant.png')

# img02 : flou
# img05 : piqué
# img04 : contour
# img10 : emboss
# img09 : verticale

# Partie B - Motif de convolution

# Identité
g = np.array([[0,0,0],[0,1,0],[0,0,0]])  # identité

# Flou
# g = 1/9*np.array([[1,1,1],[1,1,1],[1,1,1]])  # flou 3x3
# g = 1/25*np.ones((5,5))  # flou 5x5
# g = 1/16*np.array([[1,2,1],[2,4,2],[1,2,1]])  # flou gaussien 3x3
# g = 1/256*np.array([[1,2,4,6,41],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,2,4,6,41]])  # flou gaussien 5x5

# Piqué
# g = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])  # piqué

# Contour
# g = 1/9*np.array([[1,0,-1],[0,0,0],[-1,0,1]])  # contour 
# g = np.array([[0,1,0],[1,-4,1],[0,1,0]])  # contour
# g = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])  # contour

g = np.array([[-2,-1,0],[-1,1,1],[0,1,2]])  #  'emboss'

# g = np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]])  # verticale
# g = np.array([[-1,2,-1],[0,0,0],[-1,2,-1]])  # verticale (Sobel)
# g = np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]])  # horizontale
# g = np.array([[-1,-1,2],[-1,2,-1],[2,-1,-1]])  # 45 degrés

# Partie C - Calcul de la convolution
from scipy import signal
h = signal.convolve2d(f, g, boundary='fill', mode='same')


# Partie D - Affichage des images avant/après
fig = plt.figure(figsize = (10,5))
ax = plt.subplot(1,2,1)
ax.set_title("Image originale")
# fig, ax = subplots(figsize=(18, 2))
ax.imshow(f, cmap='gray')
# plt.xlim(0, 200)
# plt.ylim(0, 200)

ax = plt.subplot(1,2,2)
ax.set_title("Image sortie")
plt.imshow(h, cmap='gray')

plt.tight_layout()
plt.show()


# Partie E - Sauvegarde de l'image

h = np.clip(h,0,255)   # limite les valeurs entre 0 et 255
h = h.astype(np.uint8)  # conversion en entier
imageio.imwrite('image_apres.png', h)

