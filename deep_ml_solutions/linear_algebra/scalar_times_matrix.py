def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
	rows, cols = len(matrix), len(matrix[0])
	
	result = []
	
	for i in range(rows):
		for j in range(cols):
			result.append(matrix[i][j] * scalar)
	return result 

	# return np.dot(matrix, scalar)
