import nltk
from nltk.corpus import reuters
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt


# Où sont les données ?
# (elles doivent avoir été téléchargées avant par la commande nltk.download())
nltk.data.path.append("../../nltk_data/")


#### Un seul mot ####
def un_seul_mot(verbose=False):
    """ Analyse des occurrences de mots un par un 
    Ici mots sont en minuscules, uniquement ascii et de longueur > 1 """

    occcur = defaultdict(lambda: 0)   # dictionary occur[mot] = nb_occurrences, par défaut valeur 0

    # Occurrence de chaque mot
    for sentence in reuters.sents():
       for word in sentence:            
            if len(word) > 1 and word.isalpha() and word.isascii():
                occcur[word.lower()] += 1

    if verbose:
        print("== Stat pour les mots ==")
        NN = sum(occcur.values())  # 1 273 479
        print("Nombre total de mots ", NN)

        N = len(occcur)
        print("Nombre de mots différents ", N) # 29146

        # mot = "the"   # 69277
        # print("Nombre d'occurences de '", mot, "' :", occcur[mot]) 

        print("Mots les plus fréquents : ", Counter(occcur).most_common(10))
    
    return occcur

# Test
# occcur = un_seul_mot(verbose=False)

#### Deux mots ####

def deux_mots(verbose=False):
    """ Analyse des occurrences de mots par paires """
    # Dictionnaire des occurrence de n-mots
    occur = defaultdict(lambda: defaultdict(lambda: 0))   # dictionary occur[mot1][mot2] = nb_occurrences, par défaut valeur 0

    # Occurrence de chaque mot
    for sentence in reuters.sents():
        for w1, w2 in nltk.bigrams(sentence, pad_right=True, pad_left=True):
            if (w1 and len(w1) > 1 and w1.isalpha() and w1.isascii()) and (w2 and len(w2) > 1 and w2.isalpha() and w2.isascii()):
                occur[w1.lower()][w2.lower()] += 1

    if verbose:
        print("== Stat pour les paires de mots ==")
        som = sum([len(occur[w]) for w in occur])
        print("Nombre de paires de mots différentes :", som) # 

        # print(occur["of"]["of"]) 
        print("Mots les plus fréquents après 'of': ", Counter(occur["of"]).most_common(10))

    return occur

# Test
# occur = deux_mots(verbose=True)


###############################################
# Réalisation des probabilités conditionnelles via les co-occurrences


def probabilites(verbose=True):
    """ Calcul des probabilités conditionnelles à partir des co-occurrences 
    Génère les fichiers vocab.npy et proba.npy qui seront utilisés par my_embed_02.py """

    N = 500  # taille de notre vocabulaire : nombre de mots les plus fréquents retenus
    occur1 = un_seul_mot()   # tous les mots avec leurs occurrences
    vocab = [w for w, _ in Counter(occur1).most_common(N)] # liste des N mots les plus fréquents

    # Recherche des paires de mots consécutifs
    occur2 = deux_mots()  # toutes les paires de mots consécutifs

    cooccurrence = np.zeros((N, N))  # matrice de co-occurrence

    for i, w1 in enumerate(vocab):
        for j, w2 in enumerate(vocab):
            if w2 in occur2[w1]:
                cooccurrence[j, i] = occur2[w1][w2]


    # On initialise à 1 sur la diagonale pour être sur que les probabilités ne soient pas nulles
    np.fill_diagonal(cooccurrence, 1)

    # Normalisation pour obtenir les probabilités               
    proba = cooccurrence / cooccurrence.sum(axis=0, keepdims=True)

    # Sauvegarde des données pour éviter de les recalculer
    np.save("vocab.npy", vocab)
    np.save("proba.npy", proba)

    if verbose:
        print("== Probabilités conditionnelles ==")
        print("  Taille du vocabulaire : ", N)
        print("  Taille de la matrices des proba : ", proba.shape)
        print("  Fichiers vocab.npy et proba.npy générés")
        print("  Les premières colonnes de la matrices correspondent aux mots les plus fréquents : \n", vocab[0:10])
        print("  Vous pouvez maintenant exécuter my_embed_02.py !")


        # print("Probabilités : ", proba[0:10, 0:10])
        # print("Is nan?", np.isnan(proba).any())

    return

# Exécution indispensable une fois avant de passer à my_embed_02.py
probabilites(verbose=True)



def exemple_simple():
    """ Exemple minimaliste pour le cours """

    N = 500  # taille de notre vocabulaire
    occur1 = un_seul_mot()   # tous les mots avec leurs occurrences
    vocab = [w for w, _ in Counter(occur1).most_common(N)] # liste des N mots les plus fréquents

    # Recherche des paires de mots consécutifs
    occur2 = deux_mots()  # toutes les paires de mots consécutifs

    cooccurrence = np.zeros((N, N), dtype=np.int64)  # matrice de co-occurrence

    for i, w1 in enumerate(vocab):
        for j, w2 in enumerate(vocab):
            if w2 in occur2[w1]:
                cooccurrence[j, i] = occur2[w1][w2]


    # On initialise à 1 sur la diagonale pour être sur que les probabilités ne soient pas nulles
    # np.fill_diagonal(cooccurrence, 1)

    # Normalisation pour obtenir les probabilités               
    proba = cooccurrence / cooccurrence.sum(axis=0, keepdims=True)

    np.set_printoptions(precision=5)
    print("== Probabilités conditionnelles sur un exemple simple ==")
    print("  Taille du vocabulaire : ", N)
    print("  Vocabulaire : ", vocab)
    print("  Matrice de co-occurrence : \n", cooccurrence[0:5,0:5])
    print("  Somme des co-occurrences : \n", cooccurrence.sum(axis=0)[0:5])
    print("  Matrice des proba : \n", proba[0:5,0:5])
    print("  Somme des proba : \n", proba.sum(axis=0)[0:5])

    print("  Taille de la matrices des proba : ", proba.shape)

    return


# exemple_simple()