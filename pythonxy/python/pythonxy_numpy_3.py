
import numpy as np

# ----------------------------------------------------
# 1. Ses fonctions

print("\n\n--- 1. Ses fonctions ---\n")

def ma_formule(x):
	return np.cos(x)**2 + np.sin(x)**2

X = np.linspace(0,2*np.pi,num=20)
Y = ma_formule(X)
print(X)
print(Y)


f = lambda n:n*(n+1)/2
X = np.arange(0,10)
Y =f(X)
print(Y)

# ----------------------------------------------------
# 2. Ses fonctions (suite)

print("\n\n--- 2. Ses fonctions (suites) ---\n")

# Fonction qui n'agit que sur un élément
def valeur_absolue(x):
	if x >= 0:
		y = x
	else:
		y = -x
	return y

# Vectorisation de la fonction
vec_valeur_absolue = np.vectorize(valeur_absolue)
# vec_valeur_absolue = np.vectorize(valeur_absolue, otypes=[np.float64])

X = np.arange(-10,11,0.5) 
# X = np.array([[1,-2,3],[-4,-5,6]])
# Y = valeur_absolue(X)   # ERREUR
Y = vec_valeur_absolue(X)
print(X)
print(Y) 


# ----------------------------------------------------
# 3. Le zéro et l'infini et plus encore

print("\n\n--- 3. Le zéro et l'infini et plus encore ---\n")

X = np.arange(-1,4)
print(X)
print(1/X)

Y = np.log(X)    # -inf
Z = np.exp(Y)
print(Y)
print(Z)

X = np.array([-1,0,1,2,3,4])
print(np.log(X))     # 'nan' = Not A Number


# ----------------------------------------------------
# 4. Utilisation comme une liste

print("\n\n--- 4. Utilisation comme une liste ---\n")

# Extraire des sous-vecteurs (slicing)
X = np.linspace(0,10,num=100)
print(X[50])     # élément de rank 50
print(X[-1])     # dernier éléméent
print(X[10:20])  # éléments de rang 10 à 19
print(X[:10])    # éléments  de rang début à 9
print(X[90:])    # éléments de rang 90 jusqu'à la fin

# Ajouter un élément à un vecteur (c'est trop l'esprit de numpy)
X = np.arange(0,5,0.5)
Y = np.append(X,8.5)
print(X)
print(Y)

# Repasser à une liste
X = np.arange(0,1,0.2)
liste_X = list(X)
print(liste_X)
print(type(liste_X))


