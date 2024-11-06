import numpy as np

def shuffle_data(X, y, seed=None):
	if seed:
		np.random.seed(seed) #set seed
	indices = np.arange(X.shape[0])
	np.random.shuffle(indices)
	return X[indices], y[indices]
