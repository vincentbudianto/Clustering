import numpy as np
from numpy.matlib import repmat, repeat

'''
  agglomerative class
  agglomerative class as a whole
  - n_clusters : The number of clusters needed
'''
class agglomerative:
  # Constructor
  def __init__(self, n_clusters=1):
    self.cluster = n_clusters
  
  # Distance function (euclidean)
  def createDistanceMatrix(self, data):
    dataCount = len(data)
    distMat = np.sqrt(np.sum((repmat(data, dataCount, 1) - repeat(data, dataCount, axis=0))**2, axis=1))
    return distMat.reshape((dataCount, dataCount))
  
# TESTING
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import fowlkes_mallows_score

iris = datasets.load_iris()
data = iris.data
target = iris.target

clusters = agglomerative(n_clusters=3)
print(clusters.createDistanceMatrix(data))
print(data)