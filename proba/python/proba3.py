
import numpy as np

X = np.array([172,165,187,181,167,184,168,174,180,186])

mu = X.mean()
sigma = X.std()

print('esperance : mu =',mu)
print('écart-type : sigma =',sigma)
print('variance : sigma^2 =',sigma**2)
print('intervalle 95% :', mu-2*sigma, mu+2*sigma)

XX = (X-mu)/sigma
print('Données centrées-réduites :', XX)