#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 08:59:15 2019

@author: mahya
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM

apple_trainig_complete = pd.read_csv('AAPL_train.csv')
apple_training_processed= apple_trainig_complete.iloc[:,1:2].values
#print(open_stock_price)

#plt.figure(figsize=(25,7))
#plt.plot(open_stock_price)
#plt.xlabel('Date')
#plt.ylabel('Open Stock Price')
#plt.grid(True)
#plt.legend('Open')
#plt.show()

scaler = MinMaxScaler(feature_range=(0,1))
apple_training_scaled = scaler.fit_transform(open_stock_price)

features_set = []
labels = []
for i in range(60, 1259):
    features_set.append(apple_training_scaled[i-60:i, 0])
    labels.append(apple_training_scaled[i, 0])

features_set, labels = np.array(features_set), np.array(labels)
#print(features_set)
#print(labels)

features_set = np.reshape(features_set,(features_set.shape[0],features_set.shape[1], 1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(features_set, labels, epochs=100, batch_size=32)

apple_testing_complete = pd.read_csv('AAPL_test.csv')
apple_testing_processed = apple_testing_complete.iloc[:, 1:2].values

#plt.figure(figsize=(25,7))
#plt.plot(apple_testing_processed)
#plt.xlabel('Date')
#plt.ylabel('Open Stock Price')
#plt.grid(True)
#plt.legend('Open')
#plt.show()

apple_total = pd.concat((apple_trainig_complete['Open'], apple_testing_complete['Open']), axis=0)
test_inputs = apple_total[len(apple_total) - len(apple_testing_complete) - 60:].values
test_inputs = test_inputs.reshape(1,-1)
test_inputs = scaler.firt_transforsm(test_inputs)

test_features =[]
for i in range(60,80):
    test_features.append(test_inputs[i-60:i,0])

test_features = np.array(test_features)
test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))

predictions = model.predict(test_features)

predictions=scaler.inverse_transforms(predictions)

plt.figure(figsize=(10,6))
plt.plot(apple_testing_processed, color='blue', label='Actual Apple Stock Price')
plt.plot(predictions , color='red', label='Predicted Apple Stock Price')
plt.title('Apple Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Apple Stock Price')
plt.legend()
plt.show()













