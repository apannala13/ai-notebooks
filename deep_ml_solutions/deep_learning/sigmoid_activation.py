import math
import numpy as np 

def sigmoid(z: float) -> float:
    return np.round(1 / (1 + np.exp(-z)), 4)
