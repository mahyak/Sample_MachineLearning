# -*- coding: utf-8 -*-
"""
Created on Mon May  6 12:55:59 2019

@author: Mahya
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from scipy.stats import mode
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_sample_image

X,y = make_blobs(n_samples = 300, centers = 4, random_state = 0, cluster_std = 0.60)
plt.scatter(X[:,0], X[:,1], s = 50)

est = KMeans(4)
est.fit(X)
y_Kmeans = est.predict(X)
plt.scatter(X[:, 0], X[:, 1], c = y_Kmeans, s=50, cmap = 'rainbow')


digits = load_digits()
est = KMeans(n_clusters = 10)
clusters = est.fit_predict(digits.data)
print("10 cluster in 64 dimenstion: {}".format(est.cluster_centers_.shape))
############ visualize each of these cluter centers

fig = plt.figure(figsize = (8,3))
for i in range(10):
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    ax.imshow(est.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
    
    labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]
  
############  true cluster labels vs. K-means cluster labels    
    
X = PCA(2).fit_transform(digits.data)
kwargs = dict(cmap = plt.cm.get_cmap('rainbow', 10), edgecolor = 'none', alpha = 0.6)
fig, ax = plt.subplots(1, 2, figsize = (8, 4))
ax[0].scatter(X[:, 0], X[:, 1], c = labels, **kwargs)
ax[0].set_title('learned cluster labels')

ax[1].scatter(X[:, 0], X[:, 1], c = digits.target, **kwargs)
ax[1].set_title('true labels')

print("accuracy of K_Means classifier: {}".format(accuracy_score(digits.target, labels)))    
  
########### confusion_matrix : evaluate the accuracy of a classification
  
print(confusion_matrix(digits.target, labels))
plt.imshow(confusion_matrix(digits.target, labels), cmap = 'Blues', interpolation = 'nearest')
plt.colorbar()
plt.grid(False)
plt.xlabel('predicate')
plt.ylabel('true')
    
########## K_means for color compression  : reduce the 256^3 colors to 64.   
    
china = load_sample_image("china.jpg")
plt.imshow(china)
plt.grid(False)
print(china.shape)

image = china[::3,::3]
n_colors = 64
X = (china / 255.0).reshape(-1, 3)
model = KMeans(n_colors)
labels = model.fit_predict(X)
colors = model.cluster_centers_
new_image = colors[labels].reshape(image.shape)
new_image = (255 * new_image).astype(np.uint8)

with sns.axes_style('white'):
    plt.figure()
    plt.imshow(image)
    plt.title('input')

    plt.figure()
    plt.imshow(new_image)
    plt.title('{0} colors'.format(n_colors))

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    