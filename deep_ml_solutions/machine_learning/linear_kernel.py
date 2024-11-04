import numpy as np

def kernel_function(x1, x2):
    # K(x₁, x₂) = x₁ ⋅ x₂ = ∑(i=1 to n) x₁ᵢ ⋅ x₂ᵢ
    x1, x2 = np.array(x1), np.array(x2)
    return np.dot(x1, x2)
