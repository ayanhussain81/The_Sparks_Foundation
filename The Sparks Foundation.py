#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation - Data Science & Business Analytics Internship# 

# ## TASK 2 - Prediction using Unsupervised Machine Learning

# ## Name: Ayan Hussain

# ### AIM : From the given ‘Iris’ dataset, predict the optimum number of clusters and represent it visually

# In[1]:


# Import the required libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans


# In[2]:


iris = datasets.load_iris()
iris_data=pd.DataFrame(iris.data,columns=iris.feature_names)
iris_data.head(10)


# In[3]:


iris_data.shape


# In[4]:


iris_data.info()


# In[5]:


iris_data.describe()


# ### Finding the optimum number of clusters for k-means classification

# In[6]:


x = iris_data.iloc[:, [0, 1, 2, 3]].values
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


# ### PLOTTING RESULTS ONTO A LINE GRAPH

# In[7]:


plt.plot(range(1, 11), wcss)
plt.title('Number of clusters vs WCSS')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# ## visualizing the clusters

# In[8]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)

plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')


# ### Plotting the centroids of the clusters

# In[9]:


plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# In[10]:


plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 100, c = 'green', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 100, c = 'blue', label = 'Iris-virginica')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'black', label = 'Centroids')

plt.legend()


# ### ALL the clusters are represented by different colours.Three different colours represent three clusters. Black points represents the centroid of the cluters.
