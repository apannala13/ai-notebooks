import numpy as np

def shuffle_data(X, y, seed=None):
    if seed is not None:
        np.random.seed(seed)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    return X[indices], y[indices]
