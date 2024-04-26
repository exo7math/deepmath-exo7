
# Token
# Test avec tiktoken

import tiktoken
enc = tiktoken.get_encoding("gpt2")

# Test basique : encodage et décodage
def exemple1(phrase):
    # Encodage
    liste_tokens = enc.encode(phrase)
    print("Encodage :", phrase)
    print("Liste des tokens : ", liste_tokens)
    print([enc.decode([i]) for i in liste_tokens])

    # Décodage
    phrasebis = enc.decode(liste_tokens)
    print("Décodage :", phrasebis)

    return

# Test
exemple1("hello world")
exemple1("mathematics")
exemple1("mathematics is the queen of the sciences")


# Liste de tokens
def exemple2(nmin, nmax):
    for i in range(nmin, nmax):
        print(i, enc.decode([i]))

    return

# Test
exemple2(1000,1100)


