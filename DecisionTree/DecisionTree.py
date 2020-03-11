#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 08:28:32 2020

@author: mahya
"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from subprocess import call
from IPython.display import Image
from sklearn.tree import DecisionTreeRegressor

iris = load_iris()
X = iris.data[:, 2:]
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)

# Visualize
export_graphviz(tree_clf,
                out_file='tree.dot',
                feature_names=iris.feature_names[2:], 
                class_names=iris.target_names, 
                rounded=True,
                filled=True)

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
Image(filename = 'tree.png')

print(tree_clf.predict_proba([[5, 1.5]]))

tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X, y)