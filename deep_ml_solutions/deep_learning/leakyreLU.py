def leaky_relu(z: float, alpha: float = 0.01) -> float|int:
    if z <= 0.5:
		return alpha * z 
	else:
		return max(0, z)
