#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 08:32:32 2019

@author: mahya
"""

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

X = np.random.rand(100, 1)
y = 4 + 3 * X + np.random.rand(100, 1)


X_b = np.c_[np.ones((100, 1)), X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print(theta_best)


# Gradient Decent 

etha = 0.1
m = 100
theta = np.random.randn(2,1)
n_iteration = 1000

for iteration in range(n_iteration):
    gradient = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - etha * gradient
    
sgd_reg = SGDRegressor(max_iter=50, penalty=None, eta0=0.1)   
sgd_reg.fit(X, y.ravel())
print(sgd_reg.coef_, sgd_reg.intercept_)

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

poly_features = PolynomialFeatures(degree = 2, include_bias=False)
X_poly = poly_features.fit_transform(X)

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
print(lin_reg.coef_, lin_reg.intercept_)


def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val) 
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m])) 
        val_errors.append(mean_squared_error(y_val_predict, y_val))
        
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train") 
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)



ridge_reg =  Ridge(alpha=0.1, solver='Cholesky')
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])





































 