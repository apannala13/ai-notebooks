import numpy as np

def gini_impurity(y):
	#G = 1 - Σ(p_i^2), where p_i = count_of_class_i / num_elements_in_y
	unique_classes, counts = np.unique(y, return_counts=True)
	probas = counts / len(y)
	gini = 1 - np.sum(np.square(probas))
	return np.round(gini, 4)

	
	

	
	
	
	
