#
# Toekenisation 
# par l'algorithme BPE Byte Pair Encoding

from collections import Counter, defaultdict

# Notions 
  # corpus : le texte à tokeniser
  # mot : une suite de caractères séparé par des espaces
  # texte : une liste de mots ou token avec leur fréquence
  # token : une suite de caractères
  # vocabulaire : l'ensemble des tokens avec leur fréquence
  # paire : une suite de deux tokens
  # occurrence = fréquence : le nombre de fois qu'un élément apparait
  # fusion : la fusion de deux tokens du vocabulaire



nb_fusions = 5    # nombre de fusions (merge) à faire


def get_texte(corpus):
    """ transformation du texte en ne gardant que ascii, en minuscule, sans accents ni ponctuation
    renvoie une liste de mots sous la forme d'une suite de caractères """
    texte = corpus.lower()
    for c in texte:
        if c < 'a' or c > 'z':
            texte = texte.replace(c, " ")
    texte = texte.replace("  ", " ") 
    texte = texte.split()  # liste de mots
    texte = [ [*mots] for mots in texte ]  # liste de liste de caractères
    return texte

# Test
# print(get_texte("Le chat et le chien."))
# print(get_texte("ab ab abc dbac bacde"))


def get_occur(texte, vocab):
    """ renvoie les occurrences de chaque paire de tokens du dictionnaire texte """
    occur = defaultdict(lambda: 0)       # dictionnary occur[v1,v2] = nb_occurences
    for mot in texte:                    # pour chaque mot du texte
        for v1 in vocab:                 # pour chaque token du vocabulaire
            for v2 in vocab:             # pour chaque token du vocabulaire
                for i in range(len(mot)-1):
                    if mot[i] == v1 and mot[i+1] == v2:
                        occur[v1,v2] += 1
    return dict(occur)

# Test
# corpus = "ab ab abc dbac bacde"
# texte = get_texte(corpus)
# occur = get_occur(texte, {'a', 'b', 'c', 'd', 'e'})

# print(corpus)
# print(texte)
# print(get_occur(texte, {'a', 'b', 'c', 'd', 'e'}))


def fusion_texte(texte, v1, v2):
    """ fusionne les éléments v1 et v2 dans le texte """
    for mot in texte:
        for i in range(len(mot)-1):
            if mot[i] == v1 and mot[i+1] == v2:
                mot[i] = v1+v2
                mot.pop(i+1)
    return texte


# Test
# texte = fusion_texte(texte, 'a', 'b')
# print(texte)


def pre_tokenisation(corpus):
    " calcule des tokens par l'algorithme BPE Byte Pair Encoding "
    texte = get_texte(corpus)
    # vocabulaire initial : les caractères
    vocab =[c for mot in texte for c in mot]
    vocab = list(set(vocab))     
    
    print("Texte ", texte)
    
    # fusion des tokens les plus fréquents
    for i in range(nb_fusions):
        occur = get_occur(texte, vocab)
        print("\n Etape", i)
        print("Occurrences ", occur)
        print("Mots les plus fréquents : ", Counter(occur).most_common(20))

        count = Counter(occur).most_common(1)[0]
        print(count)
        v1, v2, occ = count[0][0], count[0][1], count[1]

        if occ == 1:   # plus rien à fusionner
            print("Plus rien à fusionner")
            break

        print("Fusion de ", v1, " et ", v2, " avec", occ, "occurrences")
        vocab.append(v1+v2)
        texte = fusion_texte(texte, v1, v2)

        print("Nouveau vocabulaire", vocab)
        print("Nouveau texte", texte)

    return vocab


# Test
print(pre_tokenisation("ab ab abc dbac bacde"))
# print(pre_tokenisation("Le chat et le chien."))


# Il resterait à faire la tokenisation proprement dite
# cad prendre un texte qcq et un vocabulaire calculé par pre_tokenisation et le transformer en une liste de tokens
# puis faire l'inverse (facile) : prendre un texte tokeniser et la transformer en un texte de mots