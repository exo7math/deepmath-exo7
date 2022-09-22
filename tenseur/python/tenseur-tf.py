
# Tenseur avec tensorflow

import tensorflow as tf
import numpy as np

# Partie A - C'est comme numpy

A = tf.constant([[1, 2], 
	             [3, 4] ])

print(A)
print(type(A))

Anp = A.numpy()  # tf 2
# sess = tf.InteractiveSession()
# Anp = A.eval()
print(Anp)
print(type(Anp))
