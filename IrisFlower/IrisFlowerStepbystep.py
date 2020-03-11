# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 11:02:10 2019

@author: Mahya
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import sklearn.tree as tree
from sklearn.externals.six import StringIO
from sklearn.ensemble import RandomForestClassifier

#iris_data = pd.read_csv('iris-data.csv', na_values = ['NA'])
##print(iris_data.head())
##print(iris_data.describe())
##
##sb.pairplot(iris_data.dropna(), hue = 'class')
#
#iris_data.loc[iris_data['class'] == 'versicolor', 'class'] = 'Iris-versicolor'
#iris_data.loc[iris_data['class'] == 'Iris-setossa', 'class'] = 'Iris-setosa'
#
##print(iris_data['class'].unique())
#
#
################## Remove failure data ##################

#iris_data = iris_data.loc[(iris_data['class'] != 'Iris-setosa') | (iris_data['sepalwidth'] >= 2.5)]
##iris_data.loc[iris_data['class'] == 'Iris-setosa', 'sepalwidth'].hist()
#
#iris_data.loc[(iris_data['class'] == 'Iris-versicolor') & (iris_data['sepallength'] <1.0)]
#iris_data.loc[(iris_data['class'] == 'Iris-versicolor') &
#              (iris_data['sepallength'] < 1.0),
#              'sepallength'] *= 100.0
##iris_data.loc[iris_data['class'] == 'Iris-versicolor', 'sepallength'].hist()
#
#iris_data.loc[(iris_data['sepallength'].isnull()) |
#              (iris_data['sepalwidth'].isnull()) |
#              (iris_data['petallength'].isnull()) |
#              (iris_data['petalwidth'].isnull())]
##iris_data.loc[iris_data['class'] == 'Iris-setosa', 'petalwidth'].hist()]
#
#average_petal_width = iris_data.loc[iris_data['class'] == 'Iris-setosa', 'petalwidth'].mean()
#
#iris_data.loc[(iris_data['class'] == 'Iris-setosamr') &
#              (iris_data['petalwidth'].isnull()),
#              'petalwidth'] = average_petal_width
#
#iris_data.loc[(iris_data['class'] == 'Iris-setosa') &
#              (iris_data['petalwidth'] == average_petal_width)]
#
#iris_data.loc[(iris_data['sepallength'].isnull()) |
#              (iris_data['sepalwidth'].isnull()) |
#              (iris_data['petallength'].isnull()) |
#              (iris_data['petalwidth'].isnull())]
#
#iris_data.to_csv('iris-data-clean.csv', index=False)
#iris_data_clean = pd.read_csv('iris-data-clean.csv')
#
##sb.pairplot(iris_data_clean, hue = 'class')
#
#
######################### Testing our data ###################

#assert len(iris_data_clean['class'].unique()) == 3
#assert iris_data_clean.loc[iris_data_clean['class'] == 'Iris-versicolor', 'sepallength'].min() >= 2.5
#assert len(iris_data_clean.loc[(iris_data_clean['sepallength'].isnull()) |
#                               (iris_data_clean['sepalwidth'].isnull()) |
#                               (iris_data_clean['petallength'].isnull()) |
#                               (iris_data_clean['petalwidth'].isnull())]) == 0

####################### Exploratery analysis ################

#plt.figure(figsize = (10,10))
#
#for column_index, column in enumerate (iris_data_clean.columns):
#    if column == 'class':
#        continue
#    plt.subplot(2, 2, column_index + 1)
#    sb.violinplot(x = 'class', y = column, data = iris_data_clean)

iris_data_clean = pd.read_csv('iris-data-clean.csv')
all_inputs = iris_data_clean[['sepallength', 'sepalwidth',
                             'petallength', 'petalwidth']].values
all_labels = iris_data_clean['class'].values
print(all_inputs[:5])

#################### Classification #################

#model_accuracies = []
#
#for repetition in range(1000):
#    (training_inputs,
#     testing_inputs,
#     training_classes,
#     testing_classes) = train_test_split(all_inputs, all_labels, test_size=0.25)
#    
#    decision_tree_classifier = DecisionTreeClassifier()
#    # Train the classifier on the training set
#    decision_tree_classifier.fit(training_inputs, training_classes)
#    # Validate the classifier on the testing set using classification accuracy
#    classifier_accuracy = decision_tree_classifier.score(testing_inputs, testing_classes)
#    model_accuracies.append(classifier_accuracy)
#    
#plt.hist(model_accuracies)


###################m 10_fold Cross validation ##########################

#def plot_cv(cv, features, labels):
#    masks = []
#    for train, test in cv.split(features, labels):
#        mask = np.zeros(len(labels), dtype=bool)
#        mask[test] = 1
#        masks.append(mask)
#    
#    plt.figure(figsize=(15, 15))
#    plt.imshow(masks, interpolation='none', cmap='gray_r')
#    plt.ylabel('Fold')
#    plt.xlabel('Row #')
#
#plot_cv(StratifiedKFold(n_splits=10), all_inputs, all_labels)
     
#################### 10_fold cross validation in our model #############

#decision_tree_classifier = DecisionTreeClassifier()
#
#cv_scores = cross_val_score(decision_tree_classifier, all_inputs, all_labels, cv = 10)
#plt.hist(cv_scores)
#plt.title('Average score: {}'.format(np.mean(cv_scores)))



####################  Grid search  ####################

decision_tree_classifier = DecisionTreeClassifier()
#parameter_grid = {'max_depth': [1, 2, 3, 4, 5],
#                  'max_features': [1, 2, 3, 4]}
#
#cross_validation = StratifiedKFold(n_splits=10)
#
#grid_search = GridSearchCV(decision_tree_classifier,
#                           param_grid=parameter_grid,
#                           cv=cross_validation)
#
#grid_search.fit(all_inputs, all_labels)
#print('Best score: {}'.format(grid_search.best_score_))
#print('Best parameters: {}'.format(grid_search.best_params_))
#grid_visualization = grid_search.cv_results_['mean_test_score']
#
#
#grid_visualization.shape = (5, 4)
#sb.heatmap(grid_visualization, cmap='Blues', annot=True)
#plt.xticks(np.arange(4) + 0.5, grid_search.param_grid['max_features'])
#plt.yticks(np.arange(5) + 0.5, grid_search.param_grid['max_depth'])
#plt.xlabel('max_features')
#plt.ylabel('max_depth')
#
#decision_tree_classifier = DecisionTreeClassifier()
#
#parameter_grid = {'criterion': ['gini', 'entropy'],
#                  'splitter': ['best', 'random'],
#                  'max_depth': [1, 2, 3, 4, 5],
#                  'max_features': [1, 2, 3, 4]}
#
#cross_validation = StratifiedKFold(n_splits=10)
#
#grid_search = GridSearchCV(decision_tree_classifier,
#                           param_grid=parameter_grid,
#                           cv=cross_validation)
#
#grid_search.fit(all_inputs, all_labels)
#print('Best score: {}'.format(grid_search.best_score_))
#print('Best parameters: {}'.format(grid_search.best_params_))
#
#
#
#with open('iris_dtc.dot', 'w') as out_file:
#    out_file = tree.export_graphviz(decision_tree_classifier, out_file=out_file)
    

##################### Random Forest Classifier #####################

random_forest_classifier = RandomForestClassifier()
parameter_grid = {'n_estimators': [10, 25, 50, 100],
                  'criterion': ['gini', 'entropy'],
                  'max_features': [1, 2, 3, 4]}
cross_validation = StratifiedKFold(n_splits=10)

grid_search = GridSearchCV(random_forest_classifier,
                           param_grid=parameter_grid,
                           cv=cross_validation)

grid_search.fit(all_inputs, all_labels)
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

grid_search.best_estimator_

random_forest_classifier = grid_search.best_estimator_

rf_df = pd.DataFrame({'accuracy': cross_val_score(random_forest_classifier, all_inputs, all_labels, cv=10),
                       'classifier': ['Random Forest'] * 10})
dt_df = pd.DataFrame({'accuracy': cross_val_score(decision_tree_classifier, all_inputs, all_labels, cv=10),
                      'classifier': ['Decision Tree'] * 10})
both_df = rf_df.append(dt_df)

sb.boxplot(x='classifier', y='accuracy', data=both_df)
sb.stripplot(x='classifier', y='accuracy', data=both_df, jitter=True, color='black')
