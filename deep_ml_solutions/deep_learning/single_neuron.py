import math
import numpy as np 

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	preds = np.dot(features, weights) + bias
	preds_sig = sigmoid(preds)
	mse = np.mean(np.square(preds_sig - labels))
	return np.round(preds_sig, 4).tolist(), np.round(mse, 4)
