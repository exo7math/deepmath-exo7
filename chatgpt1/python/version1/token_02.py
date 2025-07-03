
# Token
# Test avec transformers

from transformers import GPT2Tokenizer

# Chargement du tokenizer
enc = GPT2Tokenizer.from_pretrained("gpt2")

# print(enc("hello world"))
# print(enc("hello world")["input_ids"])

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

# Test
exemple1("hello world")
exemple1("mathematics")
exemple1("mathematics is the queen of the sciences")

exemple1(" way")
exemple1("way")
exemple1("airway")

exemple1("Man")
exemple1("man")
exemple1(" Man")
exemple1(" man")

exemple1("learning")
exemple1("surfing")

exemple1(" cat")
exemple1(" dog")


# Liste de tokens
def exemple2(nmin, nmax):
    for i in range(nmin, nmax):
        print(i, enc.decode([i]))

    return

# Test
exemple2(1000,1050)




