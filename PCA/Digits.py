# -*- coding: utf-8 -*-
"""
Created on Sun May 19 10:18:24 2019

@author: Mahya
"""

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
sns.set()

digits = load_digits()
X = digits.data
y = digits.target

pca = PCA(2) # project from 64 to 2 dimensions
Xproj = pca.fit_transform(X)
print(X.shape)
print(Xproj.shape)

plt.scatter(Xproj[:, 0], Xproj[:, 1], c = y, edgecolor = 'none', alpha = 0.5, 
            cmap=plt.cm.get_cmap('nipy_spectral', 10))
plt.colorbar()

sns.set()
pca = PCA().fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('cumulative explained variance')


fig,axes = plt.subplots(8, 8, figsize = (8,8))
fig.subplots_adjust(hspace = 0.1, wspace = 0.1)


