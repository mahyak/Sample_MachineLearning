# -*- coding: utf-8 -*-
"""
Created on Wed May  1 13:24:24 2019

@author: Mahya
"""

import pandas as pd
import seaborn as sb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score


iris_data_clean = pd.read_csv('iris-data-clean.csv')


############  we should only have three classes
assert len(iris_data_clean['class'].unique()) == 3

############ sepal lengths for 'Iris-versicolor' should never be below 2.5 cm
assert iris_data_clean.loc[iris_data_clean['class'] == 'Iris-versicolor', 'sepallength'].min() >= 2.5

############## our data set should have no missing measurements
assert len(iris_data_clean.loc[(iris_data_clean['sepallength'].isnull()) |
                               (iris_data_clean['sepalwidth'].isnull()) |
                               (iris_data_clean['petallength'].isnull()) |
                               (iris_data_clean['petalwidth'].isnull())]) == 0

all_inputs = iris_data_clean[['sepallength', 'sepalwidth',
                             'petallength', 'petalwidth']].values

all_labels = iris_data_clean['class'].values

##############   Grid Search

random_forest_classifier = RandomForestClassifier(criterion='gini', max_features=3, n_estimators=50)

##############  plot the cross-validation scores

rf_classifier_scores = cross_val_score(random_forest_classifier, all_inputs, all_labels, cv=10)
sb.boxplot(rf_classifier_scores)
sb.stripplot(rf_classifier_scores, jitter=True, color='black')

(training_inputs,
 testing_inputs,
 training_classes,
 testing_classes) = train_test_split(all_inputs, all_labels, test_size=0.25)

random_forest_classifier.fit(training_inputs, training_classes)

for input_features, prediction, actual in zip(testing_inputs[:10],
                                              random_forest_classifier.predict(testing_inputs[:10]),
                                              testing_classes[:10]):
    print('{}\t-->\t{}\t(Actual: {})'.format(input_features, prediction, actual))