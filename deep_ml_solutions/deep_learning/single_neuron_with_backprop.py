import numpy as np

def sigmoid_activation(z):
    return 1 / (1 + np.exp(-z))  # σ(z) = 1 / (1 + e^{-z})

def sigmoid_derivative(y_hat):
    return y_hat * (1 - y_hat)  # σ'(y_hat) = σ(y_hat) * (1 - σ(y_hat))
        
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
    f, s = features.shape  # (f, s) = features.shape
    weights, bias = np.array(initial_weights), np.array(initial_bias)  
    mse_hist = []
    
    for _ in range(epochs):
        z = np.dot(features, weights) + bias  # z = X * w + b
        y_hat = sigmoid_activation(z)  # ŷ = σ(z)
        mse = np.mean(np.square(y_hat - labels))  # MSE = (1/n) Σ(ŷ - y)^2
        mse_hist.append(np.round(mse, 4))  # MSE history
        
        errors = y_hat - labels  # e = ŷ - y
        sig_derivs = sigmoid_derivative(y_hat)  # σ'(z)
        wt_grad =  (2/len(labels)) * np.dot(features.T, errors * sig_derivs)  # ∇w = (2/n) X^T (e * σ'(z))
        bias_grads = (2/len(labels)) * np.sum(errors * sig_derivs)  # ∇b = (2/n) Σ(e * σ'(z))
    
        weights = weights - learning_rate * wt_grad  # w = w - η * ∇w
        bias = bias - learning_rate * bias_grads  # b = b - η * ∇b
        
    return np.round(weights, 4).tolist(), np.round(bias, 4), mse_hist 
