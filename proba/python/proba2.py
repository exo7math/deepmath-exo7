
import numpy as np

##########################################
def erreur_quadratique_moyenne(Y,Ytilde):
    n = len(Y)
    erreur = 0
    for i in range(n):
        erreur = erreur + (Y[i]-Ytilde[i])**2
    return 1/n*erreur

# Exemple
Y = np.array([1,2,5,1,3])
Ytilde = np.array([1.2,1.9,4.5,1.1,2.7])
erreur = erreur_quadratique_moyenne(Y,Ytilde)

print('Y =', Y)
print('Ytilde =', Ytilde)
print('erreur_quadratique_moyenne =', erreur)


##########################################
def erreur_absolue_moyenne(Y,Ytilde):
    n = len(Y)
    erreur = 0
    for i in range(n):
        erreur = erreur + abs(Y[i]-Ytilde[i])
    return 1/n*erreur


# Exemple 
erreur = erreur_absolue_moyenne(Y,Ytilde)
print('Y =', Y)
print('Ytilde =', Ytilde)
print('erreur_absolue_moyenne =', erreur)

##########################################
def erreur_log_moyenne(Y,Ytilde):
    n = len(Y)
    erreur = 0
    for i in range(n):
        erreur = erreur + ( np.log(Y[i]+1)-np.log(Ytilde[i]+1) )**2
    return 1/n*erreur

# Exemple 
Y = np.array([1,5,10,100])
Ytilde = np.array([2,4,8,105])
erreur1 = erreur_absolue_moyenne(Y,Ytilde)
erreur2 = erreur_log_moyenne(Y,Ytilde)

print('Y =', Y)
print('Ytilde =', Ytilde)
print('erreur_absolue_moyenne =', erreur1)
print('erreur_log_moyenne =', erreur2)

# Pour cours 
Y = np.array([100])
Ytilde = np.array([105])
erreurel1 = 1/4*erreur_absolue_moyenne(Y,Ytilde)
erreurel2 = 1*4*erreur_log_moyenne(Y,Ytilde)
print('erreur_absolue_moyenne element=', erreurel1, 100*erreurel1/erreur1, "%")
print('erreur_log_moyenne element=', erreurel2, 100*erreurel2/erreur2, "%")

##########################################
def entropie_croisee_binaire(Y,Ytilde):
    epsilon = 1e-7
    n = len(Y)
    erreur = 0
    for i in range(n):
        erreur = erreur + Y[i] * np.log(Ytilde[i] + epsilon) + (1-Y[i]) * np.log(1-Ytilde[i] + epsilon)
    return -1/n*erreur


# Exemple 
Y = np.array([1,0,1,1,0,1])
Ytilde = np.array([0.9,0.2,0.8,1,0.1,0.7])
erreur = entropie_croisee_binaire(Y,Ytilde)
print('Y =', Y)
print('Ytilde =', Ytilde)
print('entroprie_croisee_binaire =', erreur)