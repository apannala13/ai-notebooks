import numpy as np 
def pca(data: np.ndarray, k: int) -> list[list[int|float]]:
    mean = np.mean(data, axis=0)
    scaled = (data - mean) / np.std(data, axis=0)
    covariance = np.cov(scaled, rowvar=False) #rowvar=False treats rows as observations, columns as features
    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    indices = np.argsort(eigenvalues)[::-1]
    
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]
    principal_components = eigenvectors[:, :k]
    
    return principal_components.tolist()

	
