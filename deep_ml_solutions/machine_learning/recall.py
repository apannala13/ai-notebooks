import numpy as np
def recall(y_true, y_pred):
    #tp / (tp + fn)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.round(tp / (tp + fn), 3) if (tp + fn) > 0 else 0.0 

