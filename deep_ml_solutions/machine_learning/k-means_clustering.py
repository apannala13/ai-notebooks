import numpy as np 

def euclidian_distance(x, y):
	return np.sqrt(np.sum(np.square(x - y), axis=1))

def k_means_clustering(points: list[tuple[float, float]], k: int, initial_centroids: list[tuple[float, float]], max_iterations: int) -> list[tuple[float, float]]:
	points = np.array(points)
	centroids = np.array(initial_centroids)
	
	for iteration in range(max_iterations):
		distances = np.array([euclidian_distance(points, centroid) for centroid in centroids])
		cluster_assignment = np.argmin(distances, axis=0) #cluster with smallest data point
		
		new_centroids = []
		for i in range(k):
			cluster_points = points[cluster_assignment == i]
			new_centroids.append(np.mean(cluster_points, axis=0) if len(cluster_points) > 0 else centroids[i])
		
		centroids = np.array(new_centroids)
		
	return [tuple(centroid) for centroid in centroids]
