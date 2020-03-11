# -*- coding: utf-8 -*-
"""
Created on Wed May  8 17:57:06 2019

@author: Mahya

married women were asked about their participation in extramarital affairs
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pylab as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score


data = sm.datasets.fair.load_pandas().data
data['affair'] = (data.affairs > 0).astype(int)

###################### women who have affairs rate their marriages lower
#print(data.groupby('affair').mean())
#
#print(data.groupby('rate_marriage').mean())
#
#data.educ.hist()
#plt.title('Histogram Of Education')
#plt.xlabel('Education Level')
#plt.ylabel('Frequency')


################### Distribution marriage rating for those having affairs vs. not having affairs

#pd.crosstab(data.rate_marriage, data.affair.astype(bool)).plot(kind = 'bar')
#
#plt.title('Marriage Rating Distribution by Affair Rating')
#plt.xlabel('Marriage Rating')
#plt.ylabel('Frequency')

################### The percentage of woman having affairs by number of years of marriage

#affair_yrs_married = pd.crosstab(data.yrs_married, data.affair.astype(bool))
#affair_yrs_married.div(affair_yrs_married.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
#plt.title('Affair Percentage by Years Married')
#plt.xlabel('Years Married')
#plt.ylabel('Percentage')

################### Prepare Data for LogisticRegression

y, X = dmatrices('affair ~ rate_marriage + age + yrs_married + children + \
                  religious + educ + C(occupation) + C(occupation_husb)',
                  data, return_type="dataframe")

#### fit column names of X

X = X.rename(columns = {'C(occupation)[T.2.0]':'occ_2',
                        'C(occupation)[T.3.0]':'occ_3',
                        'C(occupation)[T.4.0]':'occ_4',
                        'C(occupation)[T.5.0]':'occ_5',
                        'C(occupation)[T.6.0]':'occ_6',
                        'C(occupation_husb)[T.2.0]':'occ_husb_2',
                        'C(occupation_husb)[T.3.0]':'occ_husb_3',
                        'C(occupation_husb)[T.4.0]':'occ_husb_4',
                        'C(occupation_husb)[T.5.0]':'occ_husb_5',
                        'C(occupation_husb)[T.6.0]':'occ_husb_6'})
y = np.ravel(y)

################# Logistic Regression

model = LogisticRegression()
model = model.fit(X,y)
print("Accuracy: {}".format(model.score(X,y)))
print('Null error rate: {}'.format(y.mean()))

#pd.DataFrame(zip(X.columns, np.transpose(model.coef_)))


############### split training set and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
model2 = LogisticRegression()
model2.fit(X_train, y_train)
y_pred = model2.predict(X_test)
print(y_pred)

probability = model2.predict_proba(X_test)
print(probability)

print(metrics.accuracy_score(y_test, y_pred))
print metrics.roc_auc_score(y_test, probs[:, 1])








