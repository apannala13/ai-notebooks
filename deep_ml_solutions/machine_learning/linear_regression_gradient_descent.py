import numpy as np
def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
	samples, features = X.shape
	lr = alpha
	
	coefs = np.zeros((features, 1))
	for _ in range(iterations):
		preds = np.dot(X, coefs)
		errors = preds - y.reshape(-1, 1)
		gradients = np.dot(X.T, errors) / samples 
		coefs = coefs - lr * gradients 
	return coefs.tolist()
		
		
	
