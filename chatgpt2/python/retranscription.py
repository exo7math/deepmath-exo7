# Retranscription (unembedding)

import numpy as np

n = 4

# Part A. Softmax/proba
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def exemple1():
    x = np.array([2.2, -1.1, 3.3, 0.44])
    print(softmax(x))
    

# exemple1()


# Part B. Moindres carrés
    
def moindres_carres(A,y):
    C = np.linalg.inv(A @ A.T)

    B =  A.T @ C
    x = B @ y
    return x

def exemple2():
    A = np.array([[1, 2, 0, -1], [1, 1, -1, 0]])
    y = np.array([2,-1])    
    print(A)
    print(A.T)
    print(A @ A.T)
    print(np.linalg.det(A @ A.T))
    print(np.linalg.inv(A @ A.T))
    print(A.T @ np.linalg.inv(A @ A.T))
          
    print(moindres_carres(A,y))

exemple2()
    