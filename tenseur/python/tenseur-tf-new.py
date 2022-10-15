
# Tenseur avec tensorflow

import tensorflow as tf
import numpy as np



# Partie A - Dérivée

# Une variable (en un seul point)
x = tf.Variable(3.)
with tf.GradientTape() as tape:
    y = x**2

tfgradient = tape.gradient(y, [x])
gradient = tfgradient[0].numpy()
print("Dérivée :", gradient)


# Une variable (en plusieurs points)
x = tf.Variable([1., 2., 3., 4.])
with tf.GradientTape() as tape:
    y = x**2

tfgradient = tape.gradient(y, [x])
gradient = tfgradient[0].numpy()
print("Dérivées :", gradient)



# Fonction personnalisée
def f(x):
    # return x**2
    return 2*tf.math.log(x) + tf.math.exp(-x)

x = tf.range(1, 4, 0.5)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = f(x)

tfgradient = tape.gradient(y, x)
gradient = [tfgradient[i].numpy() for i in range(len(tfgradient))]

print("Point x :", x.numpy())
print("Valeur y=f(x) :", y.numpy())
print("Dérivée f'(x) :", gradient)


# Affichage
import matplotlib.pyplot as plt

def affichage_derivee(f,a,b,epsilon=0.01):
    x = tf.range(a, b, epsilon)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = f(x)

    tfgradient = tape.gradient(y, x)
    gradient = [tfgradient[i].numpy() for i in range(len(tfgradient))]

    plt.plot(x, y, color='blue', label="f(x)")
    plt.plot(x, gradient, color='red', label="f'(x)")
    plt.legend()
    plt.tight_layout()
    # plt.savefig('tenseur-derivee.png', dpi=600)
    plt.show()
    return

def f(x):
    # return x**2
    return 2*tf.math.log(x) + tf.math.exp(-x)

# affichage_derivee(f, 1, 4, epsilon=0.01)




# Partie B - Gradient

# Exemple
def f(x, y):
     return x * y**2 

x, y = tf.Variable(2.), tf.Variable(3.)
with tf.GradientTape() as tape:
    z = f(x, y)

tfgradient = tape.gradient(z, [x, y])
gradient = [tfgradient[i].numpy() for i in range(len(tfgradient))]
dfx, dfy = gradient[0], gradient[1]
# print(X)
# print(Y)
print("Gradient :", dfx, dfy)
# print(gradient)


# Deux variables - Gradient sur une grille

# Grille numpy
n = 100
VX = tf.linspace(-2.0, 2.0, n)
VY = tf.linspace(-2.0, 2.0, n)
X, Y = tf.meshgrid(VX, VY)

# Fonction
def f(x,y):
    return x**2-y**2

with tf.GradientTape() as tape:
    tape.watch(X)
    tape.watch(Y)
    Z = f(X,Y) 

tfgradient = tape.gradient(Z, [X, Y])
gradient = [tfgradient[i].numpy() for i in range(len(tfgradient))]
dxZ, dyZ = gradient[0], gradient[1]
# print(X)
# print(Y)
# print(dxZ, dyZ)

# # Lignes de niveau
plt.contour(X, Y, Z)
plt.axis('equal') 

# # Gradient
step = 10  # pour éviter d'afficher toute les flèches
plt.quiver(X[::step,::step],Y[::step,::step],dxZ[::step,::step],dyZ[::step,::step])
plt.tight_layout()
# plt.savefig('tenseur-gradient.png', dpi=600)
# plt.show()

