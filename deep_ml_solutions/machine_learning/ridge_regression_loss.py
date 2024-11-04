import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
   # L(β) = (1/n) * Σ (yᵢ - ŷᵢ)² + λ * Σ βⱼ²
   # where βⱼ represents the weights
   outputs = np.dot(X, w)
   mse = np.mean(np.square(outputs - y))
   reg_loss = mse + (alpha * np.sum(w**2))
   return np.round(reg_loss, 4)
