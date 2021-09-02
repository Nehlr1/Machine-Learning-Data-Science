
"""""""""""
Clustering

"""""""""""
from sklearn.datasets import load_iris

import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()

data_iris = iris.data



"""""""""""
K-Mean Clustering

"""""""""""
from sklearn.cluster import KMeans

KMNS = KMeans(n_clusters= 3)

KMNS.fit(data_iris)

labels = KMNS.predict(data_iris)

Ctn = KMNS.cluster_centers_

plt.scatter(data_iris[:,2], data_iris[:,3], c = labels)
plt.scatter(Ctn[:,2], Ctn[:, 3], marker= 'o', color = 'red', s = 120)
plt.xlabel('Petal length in cm')
plt.ylabel('Petal width in cm')
plt.show()

KMNS.inertia_

K_Inertia = []

for i in range(1, 10):
    KMNS = KMeans(n_clusters= i, random_state=11)
    KMNS.fit(data_iris)
    K_Inertia.append(KMNS.inertia_)

plt.plot(range(1, 10), K_Inertia, color='green', marker='o')
plt.xlabel('Number of K')
plt.ylabel('Inertia')
plt.show()


"""""""""""
DBSCAN

"""""""""""

from sklearn.datasets import load_iris

import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()

data_iris = iris.data

from sklearn.cluster import DBSCAN

DMS = DBSCAN(eps=0.7, min_samples=4)

DMS.fit(data_iris)

Labels = DMS.labels_

plt.scatter(data_iris[:,2], data_iris[:,3], c = Labels)
plt.show()

"""""""""""
Hierarchical Clusteing

"""""""""""

from sklearn.datasets import load_iris

import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()

data_iris = iris.data

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

HR = linkage(data_iris, method='complete')

DnD = dendrogram(HR)

Labels = fcluster(HR, 4, criterion='distance')

plt.scatter(data_iris[:,2], data_iris[:,3], c = Labels)
plt.show()


