
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------
# 2. Tracé de fonctions


X = np.array([0,1,2,3,4,5])
Y = np.array([0,1,4,1,3,2])

# Au choix :
# plt.plot(X,Y)  # Segments
plt.plot(X,Y, linewidth=3, color='red')
# plt.plot(X,Y, marker='o', color='red')
# plt.scatter(X,Y)

# plt.savefig('pythonx-plot3.png')
plt.show()





