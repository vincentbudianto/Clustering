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
  def __init__(self, n_clusters=1, linkage="complete"):
    self.cluster = n_clusters
    self.linkage = linkage

  # Distance function (euclidean)
  def distance(self, data, target):
    return np.sqrt(np.sum((data - target)**2, axis=0))

  # Get maximum distance from each cluster (complete link)
  def completeLingkageDistance(self, data, target):
    maxDistance = 0
    for i in range(len(data)):
      for j in range(len(target)):
        currDistance = self.distance(data[i], target[j])
        if (currDistance > maxDistance):
          maxDistance = currDistance
    return maxDistance

  # Get the minimum distance from each cluster (single link)
  def singleLinkageDistance(self, data, target):
    minDistance = 0
    for i in range(len(data)):
      for j in range(len(target)):
        currDistance = self.distance(data[i], target[j])
        if (currDistance < minDistance):
          minDistance = currDistance
    return minDistance

  # Distance function (euclidean)
  def createDistanceMatrix(self, data):
    resultArray = []
    for i in range(len(data)):
      tempArray = []
      for j in range(i):
        tempArray.append(0)
      for j in range(i, len(data)):
        if (j == i):
          tempArray.append(0)
        else:
          tempArray.append(self.completeLingkageDistance(data[i], data[j]))
      resultArray.append(tempArray)
    return resultArray
  
  # Find minimum distance in matrix
  def findMinimumDistance(self, distMat):
    minDistance = float("inf")
    idxI = -1
    idxJ = -1
    for i in range(len(distMat)):
      for j in range((i + 1), len(distMat)):
        if distMat[i][j] < minDistance:
          minDistance = distMat[i][j]
          idxI, idxJ = i, j
    return idxI, idxJ
  
  # Merging elements
  def mergeElements(self, i, j):
    self.centroids[i].extend(self.centroids[j])
    self.labels[i].extend(self.labels[j])
    del self.centroids[j]
    del self.labels[j]
  
  # Fit function
  def fit(self, data):
    # Initialize centroids and labels
    self.centroids = []
    self.labels = []
    self.resultLabels = np.zeros(len(data))
    for i in range(len(data)):
      self.centroids.append([data[i]])
      self.labels.append([i])

    self.initial_centroids = self.centroids
    self.prev_label = None

    # Executing learning
    while(len(self.centroids) > self.cluster):
      self.prev_label = self.labels
      distMat = self.createDistanceMatrix(self.centroids)
      idxI, idxJ = self.findMinimumDistance(distMat)
      self.mergeElements(idxI, idxJ)
    
    # Convert labels to result labels
    # i = label clusters
    # j = elements in the label clusters
    for i in range(self.cluster):
      for j in range(len(self.centroids[i])):
        self.resultLabels[self.labels[i][j]] = i
    
    return self