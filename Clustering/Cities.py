#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""Importing the dataset"""
 
#Importing the dataset
dataset_cities = pd.read_csv('cities.csv')  
X_cities = dataset_cities.iloc[:, [1,2,3]].values

"""Using the elbow method to find the optimal number of clusters
"""

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss_cities = []
for i in range (1, 11):
    kmeans_cities = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans_cities.fit(X_cities)
    wcss_cities.append(kmeans_cities.inertia_)
plt.plot(range (1, 11), wcss_cities)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans_cities = KMeans(n_clusters = 2, init = 'k-means++', random_state = 0)
y_kmeans_cities = kmeans_cities.fit_predict(X_cities)

# Visualizing the clusters
# Visualizing the clusters
plt.scatter(X_cities[y_kmeans_cities == 0, 0] , X_cities[y_kmeans_cities == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X_cities[y_kmeans_cities == 1, 0] , X_cities[y_kmeans_cities == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X_cities[y_kmeans_cities == 2, 0] , X_cities[y_kmeans_cities == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X_cities[y_kmeans_cities == 3, 0] , X_cities[y_kmeans_cities == 3, 1], s = 100, c = 'green', label = 'Cluster 4')
plt.scatter(X_cities[y_kmeans_cities == 4, 0] , X_cities[y_kmeans_cities == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans_cities.cluster_centers_[:,0], kmeans_cities.cluster_centers_[:, 1], s = 300,c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel(' Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

#Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram_obj_cities = sch.dendrogram(sch.linkage(X_cities, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()


#Training the Hierarchical Clustering model on the dataset
from sklearn.cluster import AgglomerativeClustering
hc_cities = AgglomerativeClustering (n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc_cities = hc_cities.fit_predict(X_cities)
print(y_hc_cities)


# Visualizing the clusters
plt.scatter(X_cities[y_hc_cities == 0, 0] , X_cities[y_hc_cities == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X_cities[y_hc_cities == 1, 0] , X_cities[y_hc_cities == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X_cities[y_hc_cities == 2, 0] , X_cities[y_hc_cities == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X_cities[y_hc_cities == 3, 0] , X_cities[y_hc_cities == 3, 1], s = 100, c = 'green', label = 'Cluster 4')
plt.scatter(X_cities[y_hc_cities == 4, 0] , X_cities[y_hc_cities == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers (Agglomerative)')
plt.xlabel(' Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# Visualizing the clusters
from sklearn.neighbors import NearestCentroid
clf = NearestCentroid()
clf.fit(X_cities, y_hc_cities)

# Visualizing the clusters
plt.scatter(X_cities[y_hc_cities == 0, 0] , X_cities[y_hc_cities == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X_cities[y_hc_cities == 1, 0] , X_cities[y_hc_cities == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X_cities[y_hc_cities == 2, 0] , X_cities[y_hc_cities == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X_cities[y_hc_cities == 3, 0] , X_cities[y_hc_cities == 3, 1], s = 100, c = 'green', label = 'Cluster 4')
plt.scatter(X_cities[y_hc_cities == 4, 0] , X_cities[y_hc_cities == 4, 1], s = 100, c = 'green', label = 'Cluster 4')
plt.scatter(clf.centroids_[:, 0], clf.centroids_[:, 1], s = 300, c = 'yellow', label = 'Centroids') 
plt.title('Clusters of customers (Agglomerative)')
plt.xlabel(' Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
