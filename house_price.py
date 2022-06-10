# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 09:57:05 2021

@author: MarkeevP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_full = pd.read_csv('train.csv', index_col=0)
test_full = pd.read_csv('test.csv', index_col=0)

labels = train_full['SalePrice']
train_full = train_full.drop(['SalePrice'], axis=1)
# train_full.drop('Id', axis = 1, inplace=True)
# test_full.drop('Id', axis = 1, inplace=True)

# train_full = train_full.fillna('ffill', axis=1)
# train_full = train_full.fillna('bfill', axis=1)

# test_full = test_full.fillna('ffill', axis=1)
# test_full = test_full.fillna('bfill', axis=1)

# get rid of columns with a lot of nans for now
#many_nan = ['Alley','FireplaceQu','PoolQC','Fence','MiscFeature']

#categorical integers 'MSSubClass', 'OverallQual', 'OverallCond', 'BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',
# 'KitchenAbvGr', 'YrSold','MoSold','GarageCars','Fireplaces','TotRmsAbvGrd','BsmtFullBath'
cat = ['MSSubClass', 'OverallQual', 'OverallCond', 'BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',
       'KitchenAbvGr', 'YrSold','MoSold','GarageCars','Fireplaces','TotRmsAbvGrd','BsmtFullBath']


train_num1 = train_full.select_dtypes(['float64', 'int64'])
train_num = train_num1.drop(cat, axis=1)
train_num = train_num.fillna(method='ffill',axis=1)
train_num = train_num.fillna(method='bfill',axis=1)

test_num1 = test_full.select_dtypes(['float64', 'int64'])
test_num = test_num1.drop(cat, axis=1)
test_num = test_num.fillna(method='ffill',axis=1)
test_num = test_num.fillna(method='bfill',axis=1)

train_obj = train_full.select_dtypes(['object'])
test_obj = test_full.select_dtypes(['object'])

train_cat_int = train_full[cat]
test_cat_int = test_full[cat]

train_cat = pd.concat([train_obj, train_cat_int], axis=1)
test_cat = pd.concat([test_obj, test_cat_int], axis=1)

train_cat = train_cat.astype('category')
test_cat = test_cat.astype('category')

train_cat_dummies = pd.get_dummies(train_cat)
test_cat_dummies = pd.get_dummies(test_cat)

#train_num = train_num.dropna()
#print(train_full.info())
# for col in train_num.columns:
#     if len(train_num[col].unique())<50:
#         print(col, '---', train_num[col].unique())
#%%




from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()

train_num_sc = scaler.fit_transform(train_num)
test_num_sc = scaler.fit_transform(test_num)

train_num_sc_df = pd.DataFrame(train_num_sc, columns=train_num.columns)
test_num_sc_df = pd.DataFrame(test_num_sc, columns=test_num.columns)

#labels = train_full['SalePrice']

pca=PCA(4)
pca.fit(train_num_sc)
#exp_variance = pca.explained_variance_ratio_
pca_train_num = pca.transform(train_num_sc)
pca_test_num = pca.transform(test_num_sc)

# # plot the explained variance using a barplot
# fig, ax = plt.subplots()
# ax.bar( train_num.columns, exp_variance)
# plt.xticks(rotation='vertical')
# ax.set_xlabel('Principal Component #')


pca_train_num_df = pd.DataFrame(pca_train_num, columns=range(4), index = train_full.index )
pca_test_num_df = pd.DataFrame(pca_test_num, columns=range(4), index = test_full.index )

train_all_scaled = pd.concat([pca_train_num_df, train_cat_dummies], axis=1)
train_all_scaled = train_all_scaled.astype('float')

test_all_scaled = pd.concat([pca_test_num_df, test_cat_dummies], axis=1)
test_all_scaled = test_all_scaled.astype('float')


# train_all_scaled = train_all_scaled.fillna('ffill', axis=1)
# train_all_scaled = train_all_scaled.fillna('bfill', axis=1)

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

train_features, test_features, train_labels, test_labels = train_test_split(train_all_scaled, labels)
tree = DecisionTreeClassifier()
#tree.fit(train_features, train_labels)

from sklearn.ensemble import GradientBoostingRegressor
grad = GradientBoostingRegressor(max_depth=3, n_estimators=100)
grad.fit(train_all_scaled, labels)
pred = grad.predict(test_all_scaled)

from sklearn.metrics import explained_variance_score
print(explained_variance_score(test_labels, pred))

#%%
tr_cols = list(train_all_scaled.columns)
ts_cols = list(test_all_scaled.columns)
#for i in tr_cols:
#    if i == 

#%%
#  !!! put train and test together, make dummies and then devide 
train_full = pd.read_csv('train.csv', index_col=0)
test_full = pd.read_csv('test.csv', index_col=0)

#get labels and drop from train set
labels = train_full['SalePrice']
train_full = train_full.drop(['SalePrice'], axis=1)

#catigorical values that are floats or int
cat = ['MSSubClass', 'OverallQual', 'OverallCond', 'BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',
       'KitchenAbvGr', 'YrSold','MoSold','GarageCars','Fireplaces','TotRmsAbvGrd','BsmtFullBath']

# merge train and test sets to have same columns after getting the dummies
full = pd.concat([train_full, test_full])
full_num1 = full.select_dtypes(['float64', 'int64'])
full_num_with_na = full_num1.drop(cat, axis=1)
full_num = full_num_with_na.copy()
#fill na with mean of each column
for col in full_num_with_na.columns:
    full_num[col] = full_num_with_na[col].fillna(full_num_with_na[col].mean())

#isolate categorical columns and get dummies
full_obj = full.select_dtypes(['object'])
full_cat_int = full[cat]
full_cat = pd.concat([full_obj, full_cat_int], axis=1)
full_cat = full_cat.astype('category')

full_cat_dummies = pd.get_dummies(full_cat)

#Standard scaler for numerical data
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
scaler = StandardScaler()
full_num_sc = scaler.fit_transform(full_num)

#run pca to decrease number of features
n_comp = 10
pca=PCA(n_comp)
pca.fit(full_num_sc)
pca_full_num = pca.transform(full_num_sc)

pca_full_num_df = pd.DataFrame(pca_full_num, columns=range(n_comp), index = full.index)

#merge categorical and numeric together to complete features DF (maybe need to be float)
full_all_scaled = pd.concat([pca_full_num_df, full_cat_dummies], axis=1)
#full_all_scaled = full_all_scaled.astype('float')

#divide back to test and train
train_full_scaled = full_all_scaled.iloc[:1460,:]
test_full_scaled = full_all_scaled.iloc[1460:,:]

train_features, test_features, train_labels, test_labels = train_test_split(train_full_scaled, labels)


#%% play with models: Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
grad = GradientBoostingRegressor(max_depth=6, n_estimators=500)
grad.fit(train_features, train_labels)
pred = grad.predict(test_full_scaled)

from sklearn.metrics import explained_variance_score
#print(explained_variance_score(test_labels, pred))

## depth 10 result 0.144 (less better)
## depth 3 - 0.163, depth 6 - 0.152, 20 - 0.193, 8 - 0.158, 

# #%% AdaBoost- doesn' work
# from sklearn.tree import DecisionTreeClassifier
# tree = DecisionTreeClassifier()

# from sklearn.ensemble import AdaBoostRegressor
# ada = AdaBoostRegressor(base_estimator=tree, n_estimators=10)
# ada.fit(train_features, train_labels)
# pred = ada.predict(test_features)

# print(explained_variance_score(test_labels, pred))


#%% Bagging
from sklearn.ensemble import BaggingClassifier
bag=BaggingClassifier(base_estimator=tree, n_estimators=100, n_jobs=-1)
bag.fit(train_full_scaled, labels)
pred = bag.predict(test_full_scaled)

print(explained_variance_score(test_labels, pred))

##results 0.215 

#%% Random forest
from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor(min_samples_split=0.12, n_estimators=1000, n_jobs=-1)
rfr.fit(train_full_scaled, labels)
pred = rfr.predict(test_full_scaled)

#print(explained_variance_score(test_labels, pred))

## result 0.194

#%% grad boost classifier - takes too long
from sklearn.ensemble import GradientBoostingClassifier
gradC = GradientBoostingClassifier( n_estimators=5)
gradC.fit(train_full_scaled, labels)
pred = gradC.predict(test_full_scaled)

#print(explained_variance_score(test_labels, pred))

#%% Get the results
from sklearn.ensemble import GradientBoostingRegressor
grad = GradientBoostingRegressor(max_depth=10, n_estimators=50)
grad.fit(train_full_scaled, labels)
pred = grad.predict(test_full_scaled)
#%%
result = pd.DataFrame({'Id':test_full.index, 'SalePrice': pred})
result.set_index('Id', inplace=True)
result.to_csv('result.csv')
print(result)
