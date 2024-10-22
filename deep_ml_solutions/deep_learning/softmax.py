import math
import numpy as np 

def softmax(scores: list[float]) -> list[float]:
  return np.round((np.exp(scores)) / np.sum(np.exp(scores)), 4)
