# Plongement de mots
# Similarité cosinus


import numpy as np

def distance_euclidienne(x, y):
    return np.sqrt(np.sum((x-y)**2))

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


# Test

def exemple1():
    """ Pour le cours """
    x = np.array([2, 3])
    y = np.array([4, 4])
    print("Exemple 1")
    print(distance_euclidienne(x, y))
    print(cosine_similarity(x, y))  
    return

exemple1()



def exemple2():
    """ Pour le cours """
    X1 = np.array([1,0,0,1])
    X2 = np.array([0,1,0,1])
    X3 = np.array([0,1,1,1])
    Y = np.array([3,0,0,2])

    print("Exemple 2")
    print("Similarité cosinus")
    print(cosine_similarity(X1, Y))
    print(cosine_similarity(X2, Y))
    print(cosine_similarity(X3, Y))
    print("Distance euclidienne")
    print(distance_euclidienne(X1, Y))
    print(distance_euclidienne(X2, Y))
    print(distance_euclidienne(X3, Y))
    


    return

exemple2()


