import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_iterations, batch_size=1, method='batch'):
	s, f = X.shape
	for _ in range(n_iterations):
		if method == 'batch':
			preds = np.dot(X, weights)
			errors = preds - y
			gradients = np.dot(X.T, errors) / s
			weights = weights - learning_rate * gradients 
		elif method == 'stochastic':
			for i in range(s):
				preds = np.dot(X[i], weights)
				errors = preds - y[i]
				gradients = 2 * X[i] * errors
				weights = weights - learning_rate * gradients
		else:
			for i in range(0, s, batch_size):
				X_batch = X[i:i + batch_size]
				y_batch = y[i:i + batch_size]
				preds = np.dot(X_batch, weights)
				errors = preds - y_batch 
				gradients = 2 * np.dot(X_batch.T, errors) / batch_size
				weights = weights - learning_rate * gradients 
	return weights.tolist()

			
