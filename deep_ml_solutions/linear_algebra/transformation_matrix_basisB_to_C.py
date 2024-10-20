import numpy as np 
def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
	#basis - set of linearly independent vectors (all coefficients = 0) in a vector space which can be used to build any other vector in the space 
	#both B and C here are basis vectors 
	B, C = np.array(B), np.array(C)
	C_inv = np.linalg.inv(C)
	P = np.dot(C_inv, B)
	return P.tolist()
	
	#vC = [C]^-1vB 
	#Transformed Basis C = Inverse(C) * coordinate vector V in basis B
