import numpy as np 
def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
	# a = np.array(a).T
	# return a 
	matrix = a 
	rows, cols = len(matrix), len(matrix[0])
	
	transposed_matrix = [[0] * rows for _ in range(cols)]
	
	for i in range(rows):
		for j in range(cols):
			transposed_matrix[j][i] = matrix[i][j]
	return transposed_matrix
	
	

	


