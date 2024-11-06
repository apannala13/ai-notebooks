import numpy as np

def l1_regularization_gradient_descent(X: np.array, y: np.array, alpha: float = 0.1, learning_rate: float = 0.01, max_iter: int = 1000, tol: float = 1e-4) -> tuple:
    # J(w, b) = (1 / (2 * n)) * Σ[i=1 to n] (y_i - (Σ[j=1 to p] X_ij * w_j + b))^2 + α * Σ[j=1 to p] |w_j|
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    for _ in range(max_iter):
        outputs = np.dot(X, weights) + bias 
        errors = y - outputs 
        wt_grad = (1 / n_samples) * np.dot(X.T, errors) + alpha * np.sign(weights)
        wt_bias = (1/n_samples) * np.sum(errors)

        weights = weights - learning_rate * wt_grad 
        bias = bias - learning_rate * bias 

        if np.linalg.norm(wt_grad, ord=1) > tol: #manhattan distance when ord=1
            break 
    
    return np.round(weights, 4), np.round(bias, 4)


