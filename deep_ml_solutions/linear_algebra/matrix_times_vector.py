def matrix_dot_vector(a:list[list[int|float]],b:list[int|float])-> list[int|float]:
  if not len(a[0]) == len(b):
		return -1 
	
	for i in a: #for each row
		value = 0 
		for j in range(len(i)): #for each element in row
			value += i[j] * b[j]
		res.append(value)
	return res

#could just do return np.matmul(a, b)
