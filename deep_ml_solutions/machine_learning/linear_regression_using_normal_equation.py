import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	X, y = np.array(X), np.array(y)
	#OLS
	coefs = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
	return np.round(coefs, 4).tolist()

