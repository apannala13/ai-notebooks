import numpy as np 
def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
	a, b = np.array(a), np.array(b)
	
	if not len(a[0]) == len(b):
		return -1 
	res = []
	
	for i in range(len(a)):
		cur = []
		for j in range(len(b[0])):
			value = 0
			for k in range(len(b)):
				value += a[i][k] * b[k][j]
			cur.append(value)
		res.append(cur)
	return res

	# return np.matmul(a, b)

