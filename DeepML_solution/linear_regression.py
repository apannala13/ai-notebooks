#linear regression - normal equation
import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
  #exclude intercept - model only fits relationship based on features provided
  X = np.array(X)
  y = np.array(y).reshape(-1, 1) # Ensure y is a 2D array (column vector)
  theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y) #(coefs)
  theta = np.round(theta, 4).tolist()
  return theta

