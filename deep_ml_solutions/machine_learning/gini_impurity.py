import numpy as np

def gini_impurity(y):
    #Gini Impurity formula: G = 1 - Σ(p_i^2), where p_i == probability of each unique class in y
    unique_classes, counts = np.unique(y, return_counts=True)
    probas = counts / len(y)  # Calculate probabilities for each class
    gini = 1 - np.sum(np.square(probas))  # Sum of squares of probabilities
    return np.round(gini, 4)
