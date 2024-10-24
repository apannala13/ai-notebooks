import numpy as np

def f_score(y_true, y_pred, beta):
    #if beta == 1: Fb = F1
    #precision = tp / (tp + fp)
    #recall = tp / (tp + fn)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    p = (tp) / (tp + fp)
    r = (tp) / (tp + fn)
    fbeta = (1 + np.square(beta)) * ((p * r) / ((np.square(beta) * p) + r))
    return np.round(fbeta, 3) 

									 
	
	

									 
	
	
	
		
