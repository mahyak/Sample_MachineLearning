#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 18:03:52 2020

@author: mahya
"""

import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import LinearSVC
from sklearn.preprocessing import PolynomialFeatures

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]
y = (iris["target"] == 2).astype(np.float64)

# svm_clf =  Pipeline((("scaler", StandardScaler()),
#                      ("linear_svc", LinearSVC(C=1, loss="hinge")), ))

# svm_clf.fit(X,y)
# print(svm_clf.predict([[5.5, 1.7]]))



polynomial_svm_clf = Pipeline((("poly_features", PolynomialFeatures(degree=3)), 
                               ("scaler", StandardScaler()),
                               ("svm_clf", LinearSVC(C=10, loss="hinge"))
                               ))

polynomial_svm_clf.fit(X, y)


