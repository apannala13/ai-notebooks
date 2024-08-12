#Batch Gradient Descent 
import numpy as np
def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    m, n = X.shape  # m is the number of training examples (rows), n is the number of features (columns)
    coefs = np.zeros((n, 1))  # Initialize coefficients (weights) to a 2D array of zeros with n rows (one for each feature) and 1 column (output)
    learning_rate = alpha  
    for _ in range(iterations):
        predictions = np.dot(X, coefs)  # Calculate the predicted values (hypothesis) by taking the dot product of X and coefs
        errors = predictions - y.reshape(-1, 1)  # Compute the difference between predicted values and actual target values
        gradients = np.dot(X.T, errors) / m  # Compute the gradient: how much each feature contributes to the error. 
                                             # X.T is the transpose of X, and the dot product with errors gives the 
                                             # sum of the gradients for each feature. Dividing by m averages the gradients.
        coefs = coefs - learning_rate * gradients  # Update the coefficients by moving in the opposite direction of the gradient
    return np.round(coefs, 4)  
