import numpy as np 
def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
    mean, std = np.mean(data, axis = 0), np.std(data, axis = 0)
    standard_scaler = (data - mean) / std #standard scaling - guassian distribution, mean of 0 std of 1
    
    min_val, max_val = np.min(data, axis=0), np.max(data, axis=0) #transforms features to a specified range, 
                                                                  #usually 0 to 1, ensures all features have the same scale, preventing features 
                                                                  #with larger ranges from dominating the model
    min_max_scaler = (data - min_val) / (max_val - min_val)
    
    return np.round(standard_scaler,4).tolist(), np.round(min_max_scaler,4).tolist()
  
