# -*- coding: utf-8 -*-
"""
Created on Sun May 19 09:29:17 2019

@author: Mahya
"""

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA

np.random.seed(1)
X = np.dot(np.random.random(size = (2,2)), np.random.normal(size = (2, 200))).T
plt.plot(X[:, 0], X[:, 1], 'o')
plt.axis('equal')


pca = PCA(n_components = 2)
pca.fit(X)
print(pca.explained_variance_)
print(pca.components_)

plt.plot(X[:, 0], X[:, 1], 'o',alpha = 0.5)

for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    plt.plot([0, v[0]], [0, v[1]], '-k', lw=3)
plt.axis('equal');

print('one vector is longer than the other, this shows that direction in the data is more important than the others.')
print('so, the second principal component could be completely ignored without much loose of information!')

clf = PCA(0.95)
X_trans = clf.fit_transform(X)
print(X.shape)
print(X_trans.shape)


X_new = clf.inverse_transform(X_trans)
plt.plot(X[:, 0], X[:, 1], 'o', alpha = 0.2)
plt.plot(X_new[:, 0], X_new[:, 1], 'ob', alpha=0.9)
plt.axis('equal');













