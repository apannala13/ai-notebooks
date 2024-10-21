import numpy as np 
def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	mean = np.mean(data, axis=0)
	std = np.std(data, axis=0)
	standard_scaler = (data - mean) / std 
	
	min_val, max_val = np.min(data, axis=0), np.max(data, axis=0)
	min_max_scaler = (data - min_val) / (max_val - min_val)
	
	return np.round(standard_scaler, 4).tolist(), np.round(min_max_scaler, 4).tolist()
