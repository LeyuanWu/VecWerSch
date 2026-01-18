import numpy as np

# WRONG
a = b = c = np.zeros(3)
a[0] = 999
print(b)  # → [999.   0.   0.]  ← Oops!

# CORRECT
a = np.zeros(3)
b = np.zeros(3)
c = np.zeros(3)
a[0] = 999
print(b)  # → [0. 0. 0.]  ← Good!