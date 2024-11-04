import numpy as np

def kernel_function(x1, x2):
	x1, x2 = np.array(x1), np.array(x2)
	return np.dot(x1, x2)
