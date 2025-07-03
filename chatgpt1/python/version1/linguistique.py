# Linguistique et statistique
#
# inspired from
# https://medium.com/analytics-vidhya/a-comprehensive-guide-to-build-your-own-language-model-in-python-5141b3917d6d
# which itself thanks code courtesy of https://nlpforhackers.io/language-models/

import nltk
from nltk.corpus import reuters
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt


# Où sont les données ?
# (elles doivent avoir été téléchargées avant par la commande nltk.download())
# nltk.download()

nltk.data.path.append("../../nltk_data/")

# The Reuters Corpus contains 10,788 news documents totaling 1.3 million words. 
def explore_reuters():
    """ Exploration du corpus Reuters """
    # print("Nombre de catégories : ", len(reuters.categories())) # 90
    # print("Nombre de documents : ", len(reuters.fileids()))     # 10788
    # print("Nombre de mots : ", len(reuters.words()))            # 1720901
    # print("Nombre de phrases : ", len(reuters.sents()))         # 54716
    # print("Nombre de mots par phrase : ", len(reuters.words())/len(reuters.sents())) # 31.4
    # print("Nombre de mots par document : ", len(reuters.words())/len(reuters.fileids())) # 159.5
    i = 0
    for sentence in reuters.sents():
        print(" ".join(sentence))
        i += 1
        if i > 10:
            break

    return

# Test
# explore_reuters()


#### Un seul mot ####
def un_seul_mot():
    """ Analyse des occurrences de mots un par par un """
    # Dictionnaire des occurrence de n-mots
    occcur = defaultdict(lambda: 0)   # dictionnary model[mot] = nb_occurrences, par défaut valeur 0

    # Occurrence de chaque mot
    for sentence in reuters.sents():
       for word in sentence:
            if len(word) > 1:
                occcur[word] += 1


    NN = sum(occcur.values())  # 1720917
    print("Nombre total de mots ", NN)

    N = len(occcur)
    print("Nombre de mots différents ", N) # 41599

    mot = "the"   # 58251
    print("Nombre d'occurrences de '", mot, "' :", occcur[mot]) 

    print("Mots les plus fréquents : ", Counter(occcur).most_common(10))
    
    # Affichage graphique des occurrences des 50 mots les plus fréquents
    # plt.figure(figsize=(15, 5))
    # nb = 50
    # plt.plot(range(nb), [x[1] for x in Counter(occcur).most_common(nb)])
    # plt.xticks(range(nb), [x[0] for x in Counter(occcur).most_common(nb)], rotation=90)
    # plt.tight_layout()
    # # plt.savefig("linguistique-01.png", dpi=600)
    # plt.show()

    # Affichage graphique des occurrences normalisé des premiers mots les plus fréquents
    plt.figure(figsize=(15, 5))
    nb = 200  # nb de mots
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(range(1,nb+1), [x[1]/NN for x in Counter(occcur).most_common(nb)])
    # loi de Zipf-Mandelbrot
    a = 0.1     # coeff à la main
    b = 1.5     # coeff à la main
    c = 0.95     # coeff à la main
    plt.plot(range(1,nb+1), [a*1/(x+b)**c for x in range(1,nb+1)])
    plt.tight_layout()
    # # plt.savefig("linguistique-02.png", dpi=600)
    # # plt.savefig("linguistique-03.png", dpi=600)
    plt.show()
    return occcur

# Test
# occcur = un_seul_mot()


#### Deux mots ####

def deux_mots():
    """ Analyse des occurrences de mots par paires """
    # Dictionnaire des occurrence de n-mots
    occur = defaultdict(lambda: defaultdict(lambda: 0))   # dictionnary occur[mot1][mot2] = nb_occurrences, par fdéfaut valeur 0

    # Occurrence de chaque mot
    for sentence in reuters.sents():
        for w1, w2 in nltk.bigrams(sentence, pad_right=True, pad_left=True):
            if (w1 and len(w1) > 1) and (w2 and len(w2) > 1):
                occur[w1][w2] += 1

    # print(occur["that"]["is"]) # 79
    print("Mots les plus fréquents après 'the': ", Counter(occur["the"]).most_common(20))

    return occur

# Test
occur2 = deux_mots()


#### Softmax et température ####
### Voir le fichier softmax.py pour les tests ###

# Fonction softmax
def softmax(X):
    """ Fonction softmax classique """
    som = sum([np.exp(x) for x in X])
    res = [np.exp(x)/som for x in X]   
    return res


# Fonction softmax avec température
def softmaxT(X, T):
    """ Fonction softmax avec température """
    som = sum([np.exp(x/T) for x in X])
    res = [np.exp(x/T)/som for x in X]   
    return res


# Fonction choix aléatoire avec poids
def aleatoire_poids(P):
    """ Fonction qui choisit aléatoirement un indice en fonction des poids
     on part du principe que la somme des poids fait 1 """
    som = sum(P)
    P = [p/som for p in P]   # pour être sûr que la somme des poids fasse 1
    r = np.random.rand()     # nombre aléatoire entre 0 et 1
    sp = 0
    for i in range(len(P)):
        sp += P[i]
        if sp > r:
            return i
    return len(P)-1   # juste au cas où pb d'arrondi à 0.99...


# Test
# print(aleatoire_poids([0.1, 0.2, 0.7]))

# Graphe de la fonction softmax pour poids (x,1-x)
def mysoft(x):
    return np.exp(x)/(np.exp(x)+np.exp(1-x))

def mysoftT(x, T):
    return np.exp(x/T)/(np.exp(x/T)+np.exp((1-x)/T))




#### Deux mots (fin) ####

def generation_phrase(occur, mot_init, nb_mots_max=15, nb_mots_choix = 10, T=1.0):
    """ Génère une phrase à partir d'un mot initial """
    phrase = [mot_init]
    continuable = True
    while continuable and len(phrase) < nb_mots_max:
        # On prend les mots les plus fréquents
        mot_prec = phrase[-1]

        # mot le plus probable
        # mot_suiv = Counter(occur[mot_prec]).most_common(1)[0][0]

        # On calcule les mots suivants les plus probables
        occur_suiv = Counter(occur[mot_prec]).most_common(nb_mots_choix)
        mots_suiv = [x[0] for x in occur_suiv]     # liste des mots suivants
        nb_occur_suiv = [x[1] for x in occur_suiv] # nombres de leurs occurences 

        if mots_suiv == []:
            continuable = False
            break


        nb_occur_suiv = [x/sum(nb_occur_suiv) for x in nb_occur_suiv]        # normalisation des poids

        # Option A : choix aléatoire pondéré direct
        # mot_suiv = mots_suiv[aleatoire_poids(nb_occur_suiv)]  

        # Option B : choix aléatoire pondéré avec softmax et température
        poids_suiv = softmaxT(nb_occur_suiv, T)    # poids avec température
        mot_suiv = mots_suiv[aleatoire_poids(poids_suiv)]  # choix aléatoire

        phrase.append(mot_suiv)

    return " ".join(phrase)


# Test
# for _ in range(5):
#     print(generation_phrase(occur2, "the", nb_mots_choix = 20, T=1.0))


#### Trois mots ####

def trois_mots():
    """ Analyse des occurences de mots par triplets """
    # Dictionnaire des occurence de n-mots
    occur = defaultdict(lambda: defaultdict(lambda: 0))   # dictionnary occur[(mot1,mot2)][mot3] = nb_occurences, par défaut valeur 0

    # Occurence de chaque mot
    for sentence in reuters.sents():
        for w1, w2, w3 in nltk.trigrams(sentence, pad_right=True, pad_left=True):
            if (w1 and len(w1) > 1) and (w2 and len(w2) > 1) and (w3 and len(w3) > 1):
                occur[(w1,w2)][w3] += 1

    print(occur[("that","is")]["the"]) # 
    print("Mots les plus fréquents après ('that','is'): ", Counter(occur[("that","is")]).most_common(20))

    return occur

# Test
# occur3 = trois_mots()


def generation_phrase2(occur, mot_init1, mot_init2, nb_mots_max=10, nb_mots_choix=10, T=0.1):
    """ Génère une phrase à partir de deux mots initiaux """
    phrase = [mot_init1, mot_init2]
    continuable = True
    while continuable and len(phrase) < nb_mots_max:
        # On prend les mots les plus fréquents
        mot_prec1 = phrase[-2]
        mot_prec2 = phrase[-1]

        # On calcule les n mots suivants les plus probables
        occur_suiv = Counter(occur[(mot_prec1, mot_prec2)]).most_common(nb_mots_choix)
        mots_suiv = [x[0] for x in occur_suiv]     # liste des mots suivants
        nb_occur_suiv = [x[1] for x in occur_suiv] # nombres de leurs occurrences 

        if mots_suiv == []:
            continuable = False
            break
        nb_occur_suiv = [x/sum(nb_occur_suiv) for x in nb_occur_suiv]        # normalisation des poids
        poids_suiv = softmaxT(nb_occur_suiv, T)    # poids avec température
        mot_suiv = mots_suiv[aleatoire_poids(poids_suiv)]  # choix aléatoire

        phrase.append(mot_suiv)

    return " ".join(phrase)

# Test
# for _ in range(5):
#     print(generation_phrase2(occur3, "that", "company", nb_mots_max=15, nb_mots_choix=20, T=1))

