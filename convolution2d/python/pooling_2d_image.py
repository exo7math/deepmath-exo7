
import numpy as np
import matplotlib.pyplot as plt


# Partie A - Importer une image comme un tableau
import imageio.v3 as imageio
f = imageio.imread('image_07.png')



# Partie B - Pooling
n, p = f.shape

k = 4   # taille du pooling
g = f.reshape(n//k,k,p//k,k)
g = g.transpose((0, 2, 1, 3))

gmax = g.max(axis=(2,3))
gmean = g.mean(axis=(2,3))


# Partie D - Affichage des images avant/après
fig = plt.figure(figsize = (10,5))
# fig = plt.figure()
ax = plt.subplot(1,3,1)
ax.set_title("Image originale")
# fig, ax = subplots(figsize=(18, 2))
ax.imshow(f, cmap='gray')
# plt.xlim(0, 200)
# plt.ylim(0, 200)

ax = plt.subplot(1,3,2)
ax.set_title("Max-pooling")
plt.imshow(gmax, cmap='gray')

ax = plt.subplot(1,3,3)
ax.set_title("Average-pooling")
plt.imshow(gmean, cmap='gray')

plt.tight_layout()
plt.show()


# Partie E - Sauvegarde de l'image

gmax = np.clip(gmax,0,255)   # limite les valeurs entre 0 et 255
gmax = gmax.astype(np.uint8)  # conversion en entier
# imageio.imwrite('image_pooling_max.png', gmax)


gmean = np.clip(gmean,0,255)   # limite les valeurs entre 0 et 255
gmean = gmean.astype(np.uint8)  # conversion en entier
# imageio.imwrite('image_pooling_average.png', gmean)