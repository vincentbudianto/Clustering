import numpy as np

'''
	agglomerative class
	agglomerative class as a whole
	- n_clusters : The number of clusters needed
'''
class agglomerative:
	# Constructor
	def __init__(self, n_clusters=1):
		self.cluster = n_clusters

	# Distance function (euclidian)
	def distance(self, point):
		return np.argmin(np.sqrt(np.sum((point - self.centroids)**2, axis=1)))\

	# Update function
	def update_centroid(self, data):
		self.centroids = np.array([np.mean(data[self.labels == i], axis=0) for i in range(self.cluster)])

	# Predict function
	def predict(self, data):
		return np.apply_along_axis(self.distance, 1, data)

  # Other fit algoritham
	def fit(self, data):
		self.centroids = data
		self.intial_centroids = self.centroids
		self.prev_label, self.labels = None, np.zeros(len(data))


	# Fit algorithm
	def fit2(self, data):
		self.centroids = data[np.random.choice(len(data), self.cluster, replace=False)]
		self.intial_centroids = self.centroids
		self.prev_label, self.labels = None, np.zeros(len(data))

		print(self.centroids)

		while not np.all(self.labels == self.prev_label) :
			self.prev_label = self.labels
			self.labels = self.predict(data)
			self.update_centroid(data)
		
		print(self.centroids)

		return self


# TESTING

from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import fowlkes_mallows_score

iris = datasets.load_iris()
data = iris.data
target = iris.target

clusters = agglomerative
print(data)