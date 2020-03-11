# -*- coding: utf-8 -*-
"""
Created on Sun May 12 12:40:15 2019

@author: Mahya
"""

import numpy as np
import pandas as pd
import pylab as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

plt.rc('figure', figsize = (10,5))
fizsize_with_subplots = (10,10)
bin_size = 10

df_train = pd.read_csv('train.csv')
#print(df_train.head())
#print(df_train.tail())
#print(df_train.dtypes)
#print(df_train.info()) #### Missing some data for Age, Cabin and embarked
#print(df_train.describe())


################## Plot a few features to get better idea ###############

#fig = plt.figure(figsize = fizsize_with_subplots)
#fig_dims = (3, 2)
#
#
#plt.subplot2grid(fig_dims, (0, 0))
#df_train['Survived'].value_counts().plot(kind = 'bar', title  = 'Death and Survival Counts')
#
#plt.subplot2grid(fig_dims, (1, 1))
#df_train['Pclass'].value_counts().plot(kind='bar', 
#                                       title='Passenger Class Counts')
#
#plt.subplot2grid(fig_dims, (1, 0))
#df_train['Sex'].value_counts().plot(kind='bar', title='Gender Counts')
#
#plt.subplot2grid(fig_dims, (0, 1))
#df_train['Embarked'].value_counts().plot(kind = 'bar', title = 'Ports of Embarkation Counts')
#
#plt.subplot2grid(fig_dims, (2, 0 ))
#df_train['Age'].hist()
#plt.title('Age Histogram')

################## Features #############################

###### Passenger Class 

#pclass_xt = pd.crosstab(df_train['Pclass'], df_train['Survived'])
#print("The number of passengers survived based on their passenger class: {}".format(pclass_xt))
#
#pclass_xt_pct = pclass_xt.div(pclass_xt.sum(1).astype(float), axis = 0)
#pclass_xt_pct.plot(kind = 'bar', stacked = True, title = 'Survival Rate by Passenger Classes')
#plt.xlabel('Passenger Class')
#plt.ylabel('Survival Rate')

###### Sex

sexes = sorted(df_train['Sex'].unique())
genders_mapping = dict(zip(sexes, range(0, len(sexes)+1)))
print(genders_mapping)

df_train['Sex_Val'] = df_train['Sex'].map(genders_mapping).astype(int)

#sex_val_xt = pd.crosstab(df_train['Sex_Val'], df_train['Survived'])
#sex_val_xt_pct = sex_val_xt.div(sex_val_xt.sum(1).astype(float), axis = 0)
#sex_val_xt_pct.plot(kind = 'bar', stacked = True, title = 'Survival Rate by Gender')

###### Passenger class and Sex

passenger_classes = sorted(df_train['Pclass'].unique())
#
#for p_class in passenger_classes: 
#    print('M: {0} {1}'.format(p_class, len(df_train[(df_train['Sex'] == 'male') & (df_train['Pclass'] == p_class)])))
#    print('F: {0} {1}'.format(p_class, len(df_train[(df_train['Sex'] == 'female') & (df_train['Pclass'] == p_class)])))
#
#females_df = df_train[df_train['Sex'] == 'female']
#females_xt = pd.crosstab(females_df['Pclass'], females_df['Survived'])
#females_xt_pct = females_xt.div(females_xt.sum(1).astype(float), axis = 0)
#females_xt_pct.plot(kind = 'bar', stacked = True, title = 'Female Survival Rate by Passenger Class')
#plt.xlabel('Passenger Class')
#plt.ylabel('Survival Rate')

#males_df = df_train[df_train['Sex'] == 'male']
#males_xt = pd.crosstab(males_df['Pclass'], males_df['Survived'])
#males_xt_pct = males_xt.div(males_xt.sum(1).astype(float), axis = 0)
#males_xt_pct.plot(kind = 'bar', stacked = True, title = 'Male Survival Rate by Passenger Class')
#plt.xlabel('Passenger Class')
#plt.ylabel('Survival Rate')

###### Embarked

embarked_locs = (df_train['Embarked'].unique().tolist())
embarked_locs_mapping = dict(zip(embarked_locs, range(0, len(embarked_locs) + 1)))
df_train['Embarked_Val'] = df_train['Embarked'].map(embarked_locs_mapping).astype(int)
#
#df_train['Embarked_Val'].hist(bins=len(embarked_locs), range=(0, 3))
#plt.title('Port of Embarkation Histogram')
#plt.xlabel('Port of Embarkation')
#plt.ylabel('Count')
#plt.show()
#
#if len(df_train[df_train['Embarked'].isnull()] > 0):
#    df_train.replace({'Embarked_Val': { embarked_locs_mapping[np.nan]:embarked_locs_mapping['S']}}, inplace = True)
#
#embarked_locs = sorted(df_train['Embarked_Val'].unique())
#print('S/0 = Southampton; C/1 = Cherbourg; Q/2 = Queenstown'.format(embarked_locs))
#
#embarked_val_xt = pd.crosstab(df_train['Embarked_Val'], df_train['Survived'])
#embarked_val_xt_pct = embarked_val_xt.div(embarked_val_xt.sum(1).astype(float), axis = 0)
#embarked_val_xt_pct.plot(kind = 'bar', stacked = True, title = 'Survival Rate by Port of Embarkation')
#plt.xlabel('Port of Embarktion')
#plt.ylabel('Survival Rate')
#
#print("'C': 1 had the highest rate of survival, why this might be the case? dig in to Sex_val and Passenger class'")
#
#fig = plt.figure(figsize=fizsize_with_subplots) 
#rows = 2
#cols = 3
#col_names = ('Sex_Val', 'Pclass')
#
#for portIdx in embarked_locs:
#    for colIdx in range(0, len(col_names)):
#        plt.subplot2grid((rows, cols), (colIdx, portIdx))
#        df_train[df_train['Embarked_Val'] == portIdx][col_names[colIdx]].value_counts().plot(kind='bar')

###### Age

df_train['AgeFill'] = df_train['Age']
df_train['AgeFill'] = df_train['AgeFill'].groupby([df_train['Sex_Val'],df_train['Pclass']]).apply(lambda x:x.fillna(x.median()))
print('Make sure AgeFill does not contain any missing value:{}'.format(len(df_train[df_train['AgeFill'].isnull()])))

#fig, axes = plt.subplots(2, 1, figsize=fizsize_with_subplots)
#df1 = df_train[df_train['Survived'] == 0]['Age']
#df2 = df_train[df_train['Survived'] == 1]['Age']
max_age = max(df_train['AgeFill'])
#axes[0].hist([df1, df2], bins = int(max_age / bin_size), range = (1, max_age), stacked = True)
#axes[0].legend(('Died', 'Survived'), loc='best')
#axes[0].set_title('Survivors by Age Groups Histogram')
#axes[0].set_xlabel('Age')
#axes[0].set_ylabel('Count')

#axes[1].scatter(df_train['Survived'], df_train['AgeFill'])
#axes[1].set_title('Survivors by Age Plot')
#axes[1].set_xlabel('Survived')
#axes[1].set_ylabel('Age')

#for pclass in passenger_classes:
#    df_train.AgeFill[df_train.Pclass == pclass].plot(kind='kde')
#plt.title('Age Density Plot by Passenger Class')
#plt.xlabel('Age')
#plt.legend(('1st Class', '2nd Class', '3rd Class'), loc='best')

#print('1. Plot the age fill histogram for survivors')
#fig = plt.figure(figsize = fizsize_with_subplots)
#fig_dims = (3, 1)
#plt.subplot2grid(fig_dims, (0, 0))
#survived_df = df_train[df_train['Survived'] == 1]
#survived_df['AgeFill'].hist(bins = int (max_age/ bin_size), range = (1, max_age))
#
#print('2. Plot the AgeFill histogram for Females')
#plt.subplot2grid(fig_dims, (1, 0))
#females_df = df_train[(df_train['Sex_Val'] == 0) & (df_train['Survived'] == 1)]
#females_df['AgeFill'].hist(bins = int(max_age/ bin_size), range = (1, max_age))
#
#print('3. Plot the AgeFill histogram for first class passengers')
#
#plt.subplot2grid(fig_dims, (2, 0))
#class1_df = df_train[(df_train['Pclass'] == 1) & (df_train['Survived'] == 1)]
#class1_df['AgeFill'].hist(bins = int(max_age/ bin_size), range = (1, max_age))

###### Family Size

df_train['FamilySize'] = df_train['Parch'] + df_train['SibSp']
#df_train['FamilySize'].hist()
#plt.title('Family Size Histogram')

#family_size = sorted(df_train['FamilySize'].unique())
#family_size_max = max(df_train['FamilySize'])
#df1 = df_train[df_train['Survived'] == 0]['FamilySize']
#df2 = df_train[df_train['Survived'] == 1]['FamilySize']
#plt.hist([df1, df2], bins = family_size_max +1, range = (0, family_size_max), stacked = True)
#plt.legend(('Died', 'Survived'), loc='best')
#plt.title('Survivors by Family Size')

##### Preparing data for Machine Learning

print(df_train.dtypes[df_train.dtypes.map(lambda x:x == 'object')])

df_train = df_train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis = 1)
df_train = df_train.drop(['Age', 'SibSp', 'Parch', 'PassengerId'], axis=1)
print(df_train)

train_data = df_train.values

###################### Clean Data #################

def clean_data(df, drop_passenger_id):

    sexes = sorted(df['Sex'].unique()) 
    genders_mapping = dict(zip(sexes, range(0, len(sexes) + 1)))
    df['Sex_Val'] = df['Sex'].map(genders_mapping).astype(int)
    
    embarked_locs = (df['Embarked'].unique().tolist())
    embarked_locs_mapping = dict(zip(embarked_locs, range(0, len(embarked_locs) + 1)))
    df['Embarked_Val'] = df['Embarked'].map(embarked_locs_mapping).astype(int)
    
    if len(df[df['Embarked'].isnull()] > 0):
       df.replace({'Embarked_Val': { embarked_locs_mapping[np.nan]:embarked_locs_mapping['S']}}, inplace = True)

    if len(df[df['Fare'].isnull()] > 0):
        avg_fare = df['Fare'].mean()
        df.replace({ None: avg_fare }, inplace=True)
    
    df['AgeFill'] = df['Age']
    df['AgeFill'] = df['AgeFill'].groupby([df['Sex_Val'], df['Pclass']]).apply(lambda x: x.fillna(x.median()))

    df['FamilySize'] = df['SibSp'] + df['Parch']

    df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
    df = df.drop(['Age', 'SibSp', 'Parch'], axis=1)
    
    if drop_passenger_id:
        df = df.drop(['PassengerId'], axis=1)
        
    if len(df[df['Fare'].isnull()] > 0):
        avg_fare = df['Fare'].mean()
        df.replace({ None: avg_fare }, inplace=True)
    
    return df

########################## Random Forest ##########################

clf = RandomForestClassifier(n_estimators = 100)
train_features = train_data[:, 1:]
train_target = train_data[:, 0]

clf = clf.fit(train_features, train_target)
score = clf.score(train_features, train_target)

print("Mean accuracy of Random Forest: {0}".format(score))

df_test = pd.read_csv('test.csv')
df_test = clean_data(df_test, drop_passenger_id=False)
test_data = df_test.values
test_x = test_data[:, 1:]
test_y = clf.predict(test_x)

df_test['Survived'] = test_y
df_test[['PassengerId', 'Survived']].to_csv('results-rf.csv', index=False)

############### Model Accuracy ########################

train_x, test_x, train_y, test_y = train_test_split(train_features, train_target,
                                                    test_size = 0.20, random_state = 0)
print(train_features.shape, train_target.shape)
print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)

clf = clf.fit(train_x, train_y)
predict_y = clf.predict(test_x)

print ("Accuracy = {}".format((accuracy_score(test_y, predict_y))))

model_score = clf.score(test_x, test_y)
print ("Model Score %.2f \n" % (model_score))

confusion_matrix = metrics.confusion_matrix(test_y, predict_y)
print('Confusion Matrix: {}'.format(confusion_matrix))

print ("          Predicted")
print ("         |  0  |  1  |")
print ("         |-----|-----|")
print ("       0 | %3d | %3d |" % (confusion_matrix[0, 0],
                                   confusion_matrix[0, 1]))
print ("Actual   |-----|-----|")
print ("       1 | %3d | %3d |" % (confusion_matrix[1, 0],
                                   confusion_matrix[1, 1]))
print ("         |-----|-----|")


print(classification_report(test_y, predict_y, target_names =['Not Survived', 'Survived']))