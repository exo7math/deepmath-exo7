# Plongement de mots avec BERT

# Embdeding avec Bert
# https://saschametzger.com/blog/what-are-tokens-vectors-and-embeddings-how-do-you-create-them



import torch
import numpy as np
from transformers import BertModel
from transformers import BertTokenizer
import matplotlib.pyplot as plt

import random
random.seed(0)

# BERT est un modèle de langage pré-entraîné sur un très grand corpus de texte
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

## Part A: Tokenization


def exemple1():
    print("== Exemple 1 ==")
    tokens = tokenizer.tokenize("This is an example of the bert tokenizer")
    print(tokens)
    # ['this', 'is', 'an', 'example', 'of', 'the', 'bert', 'token', '##izer']

    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(token_ids)
    # [2023, 2003, 2019, 2742, 1997, 1996, 14324, 19204, 17629]

    token_ids = tokenizer.encode("This is an example of the bert tokenizer")
    print(token_ids)
    # [101, 2023, 2003, 2019, 2742, 1997, 1996, 14324, 19204, 17629, 102]

    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    print(tokens)
    # ['[CLS]', 'this', 'is', 'an', 'example', 'of', 'the', 'bert', 'token', '##izer', '[SEP]']

    return

# exemple1()


def exemple2():
    """ Renvoie la liste des vrais mots complets du vocabulaire
    (que des caractères ascii et que des lettres de a à z) """

    # Explore BERT vocabulary
    # 30522 tokens
    # numérotés de 0 à 30521
    # chaque token est un mot ou une partie de mot ex : "noise', '##ium', 'bed' et aussi '##bed' 
    # un mot qui n'est pas dans le vocabulaire sera décomposé en plusieurs tokens (ou marqué comme inconnu dans de rares cas)
    
    print("== Exemple 2 ==")
    print("Nb total de tokens :", len(tokenizer.vocab.keys()))

    # 10 mots du vocabulaire
    print("10 tokens du vocabulaire :")
    print(list(tokenizer.vocab.keys())[5000:5010])

    # Dictionnaire des tokens
    # print(tokenizer.vocab)    # ('ultimatum', 29227)

    # for token in tokenizer.vocab.keys():
        # print(token)

    # Afficher tous les tokens qui sont des mots complet (sans ##)
    list_mot_token = []
    for token in tokenizer.vocab.keys():
        # si token ne contient que des lettre de a à z:
        if token.isalpha() and token.isascii() and len(token) > 1:
            list_mot_token.append(token)

    print("Token qui sont des mots complets :", len(list_mot_token))   # 21719 vrais mots 

    # En extraire 100 au hasard
    # random.shuffle(list_mot_token)
    # print(list_mot_token[0:100])

    return list_mot_token

# exemple2()


## Part B - Embedding : exploration

def exemple3():
    """ Renvoie l'embedding d'un mot """

    print("== Exemple 3 ==")
    # get the embedding vector for the word "example"
    example_token_id = tokenizer.convert_tokens_to_ids(["example"])[0]
    example_embedding = model.embeddings.word_embeddings(torch.tensor([example_token_id]))
    print("Embedding du mot 'example' (10 premiers termes sur 768):")
    print(example_embedding[:,0:10])
    print(example_embedding.shape)  # torch.Size([1, 768])

    print("Embedding du mot 'king' :")
    king_token_id = tokenizer.convert_tokens_to_ids(["king"])[0]
    king_embedding = model.embeddings.word_embeddings(torch.tensor([king_token_id]))
    print(example_embedding[:,0:10])

    print("Embedding du mot 'queen' :")
    queen_token_id = tokenizer.convert_tokens_to_ids(["queen"])[0]
    queen_embedding = model.embeddings.word_embeddings(torch.tensor([queen_token_id]))
    print(queen_embedding[:,0:10])

    cos = torch.nn.CosineSimilarity(dim=1)
    similarity = cos(king_embedding, queen_embedding)
    similarity = similarity.detach().numpy()
    print("Similarité cosinus entre 'king' et 'queen' :", similarity) # 0.6469
    print("Angle theta :", np.degrees(np.arccos(similarity))) # 50°

    return


# exemple3()


# Part C - Embedding de mots

def vector_embedding(word):
    """ Renvoie l'embedding d'un mot sous forme de vecteur numpy """
    token_id = tokenizer.convert_tokens_to_ids([word])
    # print(token_id)
    if len(token_id) > 1:
        print("Attention : le mot n'est pas dans le vocabulaire")
    token_id = token_id[0]  # un seul token
    embedding = model.embeddings.word_embeddings(torch.tensor([token_id]))
    return embedding.detach().numpy()[0]


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def exemple4():

    print("== Exemple 4 ==")
    print("taille d'un vecteur :", vector_embedding("example").shape)

    vec1 = vector_embedding("king")
    vec2 = vector_embedding("man")
    vec3 = vector_embedding("queen")
    vec4 = vector_embedding("woman")
    similarity = cosine_similarity(vec1, vec3)
    print("Similarité cosinus de 'king' et 'queen':", similarity)
    print("Angle theta :", np.degrees(np.arccos(similarity)))

    return

# exemple4()


## Part D - Mots

# Liste de mots
list_numbers = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
list_animals = ["cat", "dog", "mouse", "elephant", "lion", "tiger", "bear", "wolf", "horse", "zebra"]
list_fruits = ["apple", "orange", "banana", "pear", "grape", "strawberry", "peach", "pineapple", "mango"]
list_countries = ["france", "germany", "spain", "italy", "portugal", "belgium", "netherlands", "austria", "greece"]
list_capitals = ["paris", "berlin", "madrid", "rome", "lisbon", "brussels", "amsterdam", "vienna", "athens"]
list_cities = ["london", "tokyo", "moscow", "beijing", "washington", "ottawa", "sydney", "dublin", "stockholm"]
list_transports = ["car", "bicycle", "motorbike", "train", "bus", "plane", "boat", "ship", "truck", "helicopter"]
list_professions = ["doctor", "lawyer", "teacher", "engineer", "scientist", "artist", "writer", "actor", "singer", "dancer"]
list_royal = ["king", "queen", "prince", "princess", "duke", "duchess", "emperor", "empress", "lord", "lady"]
list_finance = ["bank", "money", "finance", "economy", "stock", "market", "invest", "investor", "fund", "capital"]
list_colors = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "brown", "black", "white"]

def true_words(list_words):
    """ Garde uniquement les mots qui sont dans le vocabulaire """
    list_true_words = []
    for word in list_words:
        if word in tokenizer.vocab.keys():
            list_true_words.append(word)
    return list_true_words


def mywords():
    # A list of words
    list_words = []
    list_words += list_numbers
    list_words += list_animals
    list_words += list_fruits
    list_words += list_countries
    list_words += list_capitals
    list_words += list_cities
    list_words += list_transports
    list_words += list_professions
    list_words += list_royal
    list_words += list_finance
    list_words = true_words(list_words)
    # print(len(list_words))  
    return list_words

list_words = mywords()    


## Part E - Projections à la main sur une droite

def projection_droite(x,u):
    """ Projection orthogonale de x sur u 
    renvoie un scalaire """
    return np.dot(x,u) / np.dot(u,u)


def exemple5():
    u = vector_embedding("animal") - vector_embedding("city")  # direction de la droite
    list_words = list_animals + list_capitals  # liste des mots à projeter
    list_words = [w for w in list_words if w in tokenizer.vocab.keys()] # garder seulement les mots dans le vocabulaire
    list_vectors = np.array([vector_embedding(w) for w in list_words])

    list_proj = [projection_droite(x, u) for x in list_vectors]  # projection de chaque vecteur sur la droite

    # liste des projections (mot, p(mot))
    list_proj = [(list_words[i], list_proj[i]) for i in range(len(list_words))]
    list_proj.sort(key=lambda x: x[1])  # tri par ordre croissant des projections

    # plt.cla()
    plt.figure(figsize=(10, 1))
    xmin, xmax = list_proj[0][1], list_proj[-1][1]
    plt.xlim(xmin-0.02, xmax+0.03)
    plt.ylim(-2, 2)
    plt.plot([xmin-0.01, xmax+0.01], [0, 0], color='gray', linestyle='-', linewidth=1)

    np.random.seed(1)
    for i in range(len(list_proj)):
        plt.scatter(list_proj[i][1], 0, s=100, alpha=1)
        y = (-1)**i*(1 + np.random.rand())  # pour éviter que les points se superposent
        plt.plot([list_proj[i][1], list_proj[i][1]], [0, y], color='gray', linestyle='-', linewidth=1)
        plt.text(list_proj[i][1], y, list_proj[i][0], ha='center', va='center', size=9)
    
    
    plt.text(xmin-0.02, 0, "city", ha='left', va='center', size=12, weight="bold")
    plt.text(xmax+0.03, 0, "animal", ha='right', va='center', size=12, weight="bold")
    plt.axis("off")   # pas de cadre
    plt.tight_layout()
    # plt.savefig("projection-droite.png", dpi=600)

    plt.show()
    
# exemple5()

## Part E - Projections à la main sur un plan


def exemple6():

    # u = vector_embedding("animal")
    # v = vector_embedding("city")
    # u = u / np.linalg.norm(u)
    # v = v / np.linalg.norm(v)

    n = 768
    u = np.zeros(n)
    u[0] = 1
    v = np.zeros(n) 
    v[2] = 1

    list_words = list_countries + list_fruits  # liste des mots à projeter
    list_words = [w for w in list_words if w in tokenizer.vocab.keys()]

    list_vectors = np.array([vector_embedding(w) for w in list_words])

    list_proj1 = [np.dot(x, u) for x in list_vectors]
    list_proj2 = [np.dot(x, v) for x in list_vectors]

    plt.figure(figsize=(10, 10))
    for i, word in enumerate(list_words):
        plt.scatter(list_proj1[i], list_proj2[i])
        plt.annotate(word, xy=(list_proj1[i], list_proj2[i]), ha='center', va='top', weight="bold", size=15)

    plt.tight_layout()
    plt.axis("equal")
    # plt.axis("off")
    plt.xticks([])
    plt.yticks([])
    # plt.savefig("projection-plan.png", dpi=600)
    plt.show()

    return

# exemple6()


## Part F - Analyse en composantes principales ##


# Analyse en composantes principales
# cad meilleure projection de l'espace des vecteurs sur un espace de dimension 2
# https://stackoverflow.com/questions/36771525/python-pca-projection-into-lower-dimensional-space

def pca(X, k=2):
    """ Renvoie la projection des données X sur les k premiers vecteurs propre.
    Renvoie aussi les infos pour pouvoir projeter un vecteur v selon cette projection selon la formule 
    p(v) = np.dot(v-mean, U)
    """
    # Centrer les données
    Xmean = X.mean(axis=0)
    XX = X - Xmean
    # Matrice de covariance
    C = np.dot(XX.T, XX) / (XX.shape[0] - 1)
    # Décomposition en valeurs propres
    d, u = np.linalg.eigh(C)
    # Tri des valeurs propres
    idx = np.argsort(d)[::-1]
    # Tri des vecteurs propres
    u = u[:, idx]
    # Projection sur les k premiers vecteurs propres
    U = u[:, :k]
    return np.dot(XX, U), Xmean, u[:, :k], d[idx]


def exemple7():
    # list_words = list_fruits + list_countries  # pca-1
    # list_words = list_animals + list_professions + list_cities  # pca-2
    list_words = list_numbers + list_professions + list_colors # pca-0
    list_words = [w for w in list_words if w in tokenizer.vocab.keys()]
    print(len(list_words))
    print(list_words)

    list_vectors = np.array([vector_embedding(w) for w in list_words])

    # Projection des vecteurs sur un espace de dimension 2
    Z, _, _, _ = pca(list_vectors, k=2)

    # Affichage 2D
    plt.figure(figsize=(10, 10))
    for i, word in enumerate(list_words):
        plt.scatter(Z[i, 0], Z[i, 1])
        plt.annotate(word, xy=(Z[i, 0], Z[i, 1]), ha='center', va='top', weight="bold", size=13)

    plt.tight_layout()
    plt.axis("equal")
    # plt.axis("off")
    plt.xticks([])
    plt.yticks([])
    plt.savefig("projection-pca-0.png", dpi=600)
    plt.show()

    return

exemple7()


def exemple8():
    # list_words = list_fruits + list_countries  # 
    list_words = list_animals + list_professions + list_cities  # pca-3

    list_vectors = np.array([vector_embedding(w) for w in list_words])

    # Projection des vecteurs sur un espace de dimension 3
    Z, _, _, _ = pca(list_vectors, k=3)

    # Affichage 3D
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    
    for i, word in enumerate(list_words):
        ax.scatter(Z[i, 0], Z[i, 1], Z[i, 2])
        ax.text(Z[i, 0], Z[i, 1], Z[i, 2], word)

    ax.view_init(azim=-35, elev=40)
    # plt.tight_layout()
    # plt.axis("off")
    # plt.xticks([])
    # plt.yticks([])
    # plt.savefig("projection-pca-3.png", dpi=600)
    plt.show()

    return

# exemple8()



def exemple9():
    """ Affichage 2D de tous les mots """

    # Tous les mots sont dans le vocabulaire
    list_words = [w for w in tokenizer.vocab.keys()]

    # numpy array of all embeddings
    list_vectors = np.array([vector_embedding(w) for w in list_words])

     # Projection des vecteurs sur un espace de dimension 2
    Z, _, _, _ = pca(list_vectors, k=2)

    # ZZ = np.dot(list_vectors-mean, U)  # projection sur les 2 premiers vecteurs propres en réutilisant la projection précédente

    plt.scatter(Z[:, 0], Z[:, 1],s=1)
    # for i, word in enumerate(list_words):
        # plt.annotate(word, xy=(ZZ[i, 0], ZZ[i, 1]))
    
    plt.tight_layout()
    plt.axis("equal")
    # plt.axis("off")
    plt.xticks([])
    plt.yticks([])
    # plt.savefig("projection-pca-4.png", dpi=600)
    plt.show()

    return

# exemple9()

## Part G - King - Man + Woman = Queen

def exemple10():
    """ Affichage 2D de King - Man + Woman = Queen """

    list_words = ["king", "man", "queen", "woman"]  # pca-5

    # list_words = list_countries + list_capitals  # pca-6

    list_vectors = np.array([vector_embedding(w) for w in list_words])

     # Projection des vecteurs sur un espace de dimension 2
    Z, _, _, _ = pca(list_vectors, k=2)


    plt.scatter(Z[:, 0], Z[:, 1],s=20)
    for i, word in enumerate(list_words):
        plt.annotate(word, xy=(Z[i, 0], Z[i, 1]), ha='center', va='top', weight="bold", size=13)
    
    plt.tight_layout()
    plt.axis("equal")
    # plt.axis("off")
    plt.xticks([])
    plt.yticks([])
    # plt.savefig("projection-pca-6.png", dpi=600)
    plt.show()

    return

# exemple10()

## Part H - Mots les plus proches

def mots_les_plus_proches_vecteur(vecteur, list_mot_token, n=10):
    # Calculer le vecteur du mot
    # Calculer la distance entre le vecteur du mot et tous les autres vecteurs
    list_dist = []
    for mot_token in list_mot_token:
        vec_mot_token = vector_embedding(mot_token)
        # dist = np.linalg.norm(vecteur - vec_mot_token)  # distance euclidienne
        dist = cosine_similarity(vecteur, vec_mot_token)  # similarité cosinus
        list_dist.append(dist)
    # Trouver le mot le plus proche
    # idx = np.argsort(list_dist)[0:n]  # indices des n plus petites distances euclidiennes
    idx = np.argsort(list_dist)[::-1][0:n]  # indices des n plus grandes similarité cosinus
    return [list_mot_token[i] for i in idx]


def mots_les_plus_proches(mot, list_mot_token, n=10):
    vecteur = vector_embedding(mot)
    return mots_les_plus_proches_vecteur(vecteur, list_mot_token, n)



def exemple11():
    """ Affichage des mots les plus proches """

    list_words = [w for w in tokenizer.vocab.keys()]    

    print("== Exemple 11 ==")
    print("Mots les plus proches de 'king' :")
    print(mots_les_plus_proches("king", list_words))
    print("Mots les plus proches de 'queen' :")
    print(mots_les_plus_proches("queen", list_words))


    vec1 = vector_embedding("king")
    vec2 = vector_embedding("man")
    vec3 = vector_embedding("woman")

    vector = vec1 - vec2 + vec3

    print("Mots les proches de 'king-man+woman' :")
    print(mots_les_plus_proches_vecteur(vector, list_words))

    return

# exemple11()



