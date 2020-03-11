# -*- coding: utf-8 -*-
"""
Created on Sat May  4 10:30:34 2019

@author: Mahya
"""

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris = load_iris()

#####################  X (feature)/ y(responce): Train/Test: same ##############

X = iris.data
y = iris.target

##################### Logestic Regression ####################

logreg = LogisticRegression()
logreg.fit(X,y)
# predict the response values for the observations in X
y_pred = logreg.predict(X)

print("The number of prediction were generated: {}".format(len(y_pred)))
# "training accuracy": train and test the model on the same data
print("Classification accuracy for logestic regression: {}".format(metrics.accuracy_score(y, y_pred)))

#################### KNN (K = 5) ##############################

KNN = KNeighborsClassifier(n_neighbors = 5)
KNN.fit(X,y)
y_pred = KNN.predict(X)
print("Classification accuracy for KNN = 5: {}".format(metrics.accuracy_score(y, y_pred)))

#################### KNN (K = 1) ############################

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X, y)
y_pred = knn.predict(X)
print("Classification accuracy for KNN = 1: {}".format(metrics.accuracy_score(y, y_pred)))


#####################  X (feature)/ y(responce): Train/Test split ##############

print("X before split: {}".format(X.shape))
print("y before split: {}".format(y.shape))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 4)
print("X tarin set after split: {}".format(X_train.shape))
print("X test set after split: {}".format(X_test.shape))
print("y train set after split: {}".format(y_train.shape))
print("y test set after split: {}".format(y_test.shape))


###################### Logestic regression ##################

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print("Logestic regression accuracy for split dataset: {}".format(metrics.accuracy_score(y_test, y_pred)))


##################### KNN (k = 5) #########################

KNN = KNeighborsClassifier(n_neighbors = 5)
KNN.fit(X_train, y_train)
y_pred = KNN.predict(X_test)

print("KNN 5 accuracy for split dataset: {}".format(metrics.accuracy_score(y_test, y_pred)))

##################### KNN (k = 1) ########################

KNN = KNeighborsClassifier(n_neighbors = 1)
KNN.fit(X_train, y_train)
y_pred = KNN.predict(X_test)

print("KNN 1 accuracy for split dataset: {}".format(metrics.accuracy_score(y_test, y_pred)))


################### Locate better value for K ################

k_range = list(range(1, 26))
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')

print('Plot analysis: Training accuracy rises as model complexity increases Testing accuracy penalizes models that are too complex or not complex enough For KNN models, complexity is determined by the value of K (lower value = more complex)')
















































