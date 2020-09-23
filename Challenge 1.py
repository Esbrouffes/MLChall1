# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Libraries: Standard ones
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Library for boxplots
import seaborn as sns

# K-means function
from sklearn.cluster import KMeans

# Functions for silhouette
from sklearn.metrics import silhouette_samples, silhouette_score

# Function to standardize the data 
from sklearn.preprocessing import scale

# Functions for hierarchical clustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist





food = pd.read_csv("foodata.csv",sep=";")
happiness=pd.read_csv("happiness_2019.csv", sep=",")
shootings=pd.read_csv("shootings.csv", sep=",")
temperat=pd.read_csv("temperat.csv", sep=",")
Rank=pd.read_csv("temperat.csv", sep=",")
#print(food)
#print(food.describe())
"""
plt.figure()
plt.title('boxplot food s features')
food_box=sns.boxplot(data=food,fliersize=10) 
plt.show()  # fliersize is the size used to indicate the outliers
plt.figure()
plt.title ('boxplot ruspini s features')
ruspini_box=sns.boxplot(data=ruspini,fliersize=10)
plt.show()

"""
plt.figure()
plt.scatter(ruspini['x'],ruspini['y'])
plt.show()
kmeans = KMeans(n_clusters=4,n_init=1,init='random').fit(ruspini)
centers=kmeans.cluster_centers_
kmeans.labels_
plt.scatter(ruspini['x'], ruspini['y'],c=kmeans.labels_)
