import numpy as np

mask = np.array([[1,2],[3,4]])

print(mask)

mask = np.array([mask for i in range(3)])
print(mask.shape)