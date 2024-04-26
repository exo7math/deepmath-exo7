# Plongement de mots avec GPT-2

from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer

import tensorflow as tf
from tensorboard.plugins import projector

import numpy as np

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

model = GPT2LMHeadModel.from_pretrained('gpt2')



def my_tokenizer(phrase):
    # Encodage
    liste_tokens = tokenizer.encode(phrase)
    print("Encodage :", phrase)
    print("Liste des tokens : ", liste_tokens)
    print([tokenizer.decode([i]) for i in liste_tokens])

    # Décodage
    phrasebis = tokenizer.decode(liste_tokens)  # si plusieurs tokens: moyenne des vecteurs
    print("Décodage :", phrasebis)

    return liste_tokens


# Test 
my_tokenizer(" man")
my_tokenizer(" woman")
my_tokenizer(" king")
my_tokenizer(" queen")

# Récupérer la matrice M de l'embedding
word_embeddings = model.transformer.wte.weight
# position_embeddings = model.transformer.wpe.weight  # Word Position Embeddings 
M = word_embeddings.detach().numpy()
print(M.shape)

def vector_embedding(word):
    token = tokenizer.encode(word)   
    if len(token) > 1:
        print("Warning: more than one token for word", word)
    all_vectors = M[token, :]
    vector = np.mean(all_vectors, axis=0)
    return vector

# Test
print(vector_embedding("hello").shape)
vec_dog = vector_embedding(" dog")
vec_cat = vector_embedding(" cat")
print(" dog", vec_dog[:10])
print(" cat", vec_cat[:10])


# Similarité cosinus
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))




# Test 
# vec1 = vector_embedding(" king")
# vec2 = vector_embedding(" man")
# vec3 = vector_embedding(" queen")
# vec4 = vector_embedding(" woman")

# print(vec1[:10])
# print(vec2[:10])

similarity = cosine_similarity(vec_dog, vec_cat)
print('Similarité cosinus :', similarity)
print('Angle theta :', np.degrees(np.arccos(similarity)))




# Recherche des tokens les plus proches d'un token donné
def closest_tokens(token, topn=10):
    vector = vector_embedding(token)
    token_id = tokenizer.encode(token)[0]
    max_similarity = -1
    max_token = -1  # token le plus proche
    N = 50257  # nombre de tokens
    liste_tokens = []
    for i in range(N):
        sim = cosine_similarity(vector, M[i, :])
        liste_tokens.append( (i, sim) )

    liste_tokens.sort(key=lambda x: x[1], reverse=True)
    
    print("Tokens les plus proches de", token)
    for i in range(topn):
        token_id, sim = liste_tokens[i]
        print(tokenizer.decode([token_id]), sim)

    print("Tokens les plus éloignés de", token)
    for i in range(topn):
        token_id, sim = liste_tokens[N-i-1]
        print(tokenizer.decode([token_id]), sim)

    
    return liste_tokens

# Test
closest_tokens(" dog")


