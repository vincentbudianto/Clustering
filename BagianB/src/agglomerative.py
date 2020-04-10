import numpy as np
import copy as cp
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
  def distance(self, data, target):
    return np.sqrt(np.sum((data - target)**2, axis=0))

  # Get maximum distance from each cluster (complete link)
  def clusterDistance(self, data, target):
    maxDistance = 0
    for i in range(len(data)):
      for j in range(len(target)):
        currDistance = self.distance(data[i], target[j])
        if (currDistance > maxDistance):
          maxDistance = currDistance
    return maxDistance

  # Distance function (euclidean)
  def createDistanceMatrix(self, data):
    resultArray = []
    for i in range(len(data)):
      tempArray = []
      for j in range(i):
        tempArray.append(0)
      for j in range(i, len(data)):
        tempArray.append(self.clusterDistance(data[i], data[j]))
      resultArray.append(tempArray)
    return resultArray
  
  # Find minimum distance in matrix
  def findMinimumDistance(self, distMat):
    i, j = np.where(distMat == np.min(distMat[np.nonzero(distMat)]))
    return (i, j)

# TESTING
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import fowlkes_mallows_score

iris = datasets.load_iris()
data = iris.data
target = iris.target

clusters = agglomerative(n_clusters=3)
clusterData = []

for i in range(len(data)):
  clusterData.append([data[i]])

distMat = clusters.createDistanceMatrix(clusterData)
print(distMat)


# TESTING
# from sklearn import datasets
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.metrics import silhouette_score
# from sklearn.metrics.cluster import fowlkes_mallows_score

# iris = datasets.load_iris()
# data = iris.data
# target = iris.target



# clusters = agglomerative(n_clusters=3)
# clusterData = cp.deepcopy(data)

# for (i in range(len(clusterData))):

# distMat = clusters.createDistanceMatrix(data)
# minCluster, minClusterMirror = clusters.findMinimumDistance(distMat)
# clusterMergedA = minCluster[0]
# clusterMergedB = minCluster[1]
# print(clusterMergedA)
# print(clusterMergedB)
