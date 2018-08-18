# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 10:20:27 2018

@author: Anuj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
import xgboost

submission = pd.DataFrame()

train = pd.read_csv('/Users/Anuj/Desktop/Python codes/Kaggle/Housing prices/train.csv')
test = pd.read_csv('/Users/Anuj/Desktop/Python codes/Kaggle/Housing prices/test.csv')

train = train[train['GarageArea'] < 1200]

num_train = train.select_dtypes(exclude = ['object']).copy()
obj_train = train.select_dtypes(include = ['object']).copy()

obj_train = pd.get_dummies(obj_train)

concat_train = pd.concat([num_train, obj_train], axis = 1, sort = False)
concat_train.reindex(sorted(concat_train.columns), axis = 1)

concat_train.to_csv('/Users/Anuj/Desktop/Python codes/Kaggle/Housing prices/train_new.csv', sep = ',')

train_new = pd.read_csv('/Users/Anuj/Desktop/Python codes/Kaggle/Housing prices/train_new_1.csv')


X = train_new.drop('SalePrice', axis = 1)
y = np.log(train_new['SalePrice'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

X_train = X_train.fillna(X_train.mean())
X_test= X_test.fillna(X_test.mean())

sc = StandardScaler()
sc_X_train = sc.fit_transform(X_train)
sc_X_test = sc.fit_transform(X_test)

n_estimators = [1500, 2000]
learning_rate = [0.07]
max_depth = [5,7]
subsample = [0.65, 0.75]

xgb = xgboost.XGBRegressor(n_estimators = 100, max_depth = 5, learning_rate = 0.07, subsample = 0.75)
grid = GridSearchCV(estimator = xgb, cv = 3, param_grid = dict(n_estimators = n_estimators, learning_rate = learning_rate, max_depth = max_depth, subsample = subsample))
xgb.fit(sc_X_train, y_train)

print(grid.best_params_)

y_pred = xgb.predict(sc_X_test)

#TESTING

num_test = test.select_dtypes(exclude = ['object']).copy()
obj_test = test.select_dtypes(include = ['object']).copy()

obj_test = pd.get_dummies(obj_test)
concat_test = pd.concat([num_test, obj_test], axis = 1, sort = False)
concat_test = concat_test.fillna(concat_test.mean())

'''
df = pd.merge(concat_train, concat_test, how='outer', indicator=True)
rows_in_df1_not_in_df2 = df[df['_merge']=='left_only'][concat_train.columns]
'''
concat_test['MiscFeature_Tenc'] = pd.Series(0, index = concat_test.index) 
concat_test['PoolQC_Fa'] = pd.Series(0, index = concat_test.index)
concat_test['GarageQual_Ex'] = pd.Series(0, index = concat_test.index)
concat_test['Electrical_Mix'] = pd.Series(0, index = concat_test.index)
concat_test['Heating_OthW'] = pd.Series(0, index = concat_test.index)
concat_test['Heating_Floor'] = pd.Series(0, index = concat_test.index)
concat_test['Exterior2nd_Other'] = pd.Series(0, index = concat_test.index)
concat_test['Exterior1st_Stone'] = pd.Series(0, index = concat_test.index)
concat_test['Exterior1st_ImStucc'] = pd.Series(0, index = concat_test.index)
concat_test['RoofMatl_Roll'] = pd.Series(0, index = concat_test.index)
concat_test['RoofMatl_Metal'] = pd.Series(0, index = concat_test.index)
concat_test['RoofMatl_Membran'] = pd.Series(0, index = concat_test.index)
concat_test['RoofMatl_ClyTile'] = pd.Series(0, index = concat_test.index)
concat_test['HouseStyle_2.5Fin'] = pd.Series(0, index = concat_test.index)
concat_test['Condition2_RRNn'] = pd.Series(0, index = concat_test.index)
concat_test['Condition2_RRAn'] = pd.Series(0, index = concat_test.index)
concat_test['Condition2_RRAe'] = pd.Series(0, index = concat_test.index)
concat_test['Utilities_NoSeWa'] = pd.Series(0, index = concat_test.index)

concat_test.reindex(sorted(concat_test.columns), axis=1)

concat_test.to_csv('/Users/Anuj/Desktop/Python codes/Kaggle/Housing prices/test_new.csv', sep = ',')

test_new = pd.read_csv('/Users/Anuj/Desktop/Python codes/Kaggle/Housing prices/test_new_1.csv')

sc_concat_test = sc.fit_transform(test_new)

y_test_pred = xgb.predict(sc_concat_test)
result = np.exp(y_test_pred)

#Saving the dataset in a CSV file

submission = pd.DataFrame()
submission['Id'] = test.Id

submission['SalePrice'] = result

print(submission.head())

submission.to_csv('/Users/Anuj/Desktop/Python codes/Kaggle/Housing prices/Submission8.csv', index=False)

