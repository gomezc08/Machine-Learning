#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""Importing the dataset"""

#Importing the dataset
dataset = pd.read_csv('USArrests.csv')
X = dataset.iloc[:,1:5].values

"""Using the elbow method to find the optimal number of clusters
"""

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range (1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(X) 
    wcss.append(kmeans.inertia_)
plt.plot(range (1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

"""Fitting K-Means to the dataset"""

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(X)

"""Visualizing the clusters"""

# Visualizing the clusters
plt.scatter(X[y_kmeans == 0, 0] , X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0] , X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0] , X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:, 1], s = 300,	c = 'yellow', label = 'Centroids')
plt.title('Clusters of US States')
plt.xlabel(' Murder')
plt.ylabel('Assault')
plt.legend()
plt.show()

#Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram_obj = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

"""Training the Hierarchical Clustering model on the dataset"""

#Training the Hierarchical Clustering model on the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering (n_clusters = 2, affinity = 'euclidean', 	linkage = 'ward')
y_hc = hc.fit_predict(X)
print(y_hc)

"""Visualizing the clusters"""

# Visualizing the clusters
plt.scatter(X[y_hc == 0, 0] , X[y_hc == 0, 1], s = 100, 
c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0] , X[y_hc == 1, 1], s = 100, 
c = 'blue', label = 'Cluster 2')
plt.title('Clusters of US States – Agglomerative clustering')
plt.xlabel('Murder')
plt.ylabel('Assault')
plt.legend()
plt.show()