# -*- coding: utf-8 -*-
"""
Created on Fri May 14 15:09:06 2021

@author: User
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
gender=pd.read_csv('gender_submission.csv')



train = train[train['Age'].notna()]
train = train[train['Embarked'].notna()]

# worse for KNN, good for others (Keep columns as is just as 0,1 categories)
# train['Nsex'] = train.Sex.replace(['female','male'],[0,1]) 
# train['Nembarked'] = train.Embarked.replace(['S', 'C', 'Q'],[0, 1, 2])
# xx = train.drop(['Survived','PassengerId', 'Name', 'Sex', 'Cabin', 'Ticket', 'Embarked'], axis=1) 

# # Get Dummies (special columns with 1) better for KNN, NB doesn't
train1 = pd.get_dummies(train, columns=['Embarked', 'Pclass', 'Sex', 'Parch', 'SibSp']) 
xx = train1.drop(['Survived','PassengerId', 'Name', 'Cabin', 'Ticket', 'Fare'], axis=1) 

X=xx.values
y=train.Survived.values

#%% devide train data to train and test
from sklearn.model_selection import train_test_split
Xtr, Xtest, ytr, ytest = train_test_split(X, y, test_size=0.3, random_state=42)
#tr for train

#%% (1) KNN K-nearest nighbors

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=30)
knn.fit(Xtr, ytr)
y_pred = knn.predict(Xtest)
print('KNN accuracy - %.4f'%knn.score(Xtest, ytest))

# Result around 68-70% for 20 neighbors

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error as MSE
cv = - cross_val_score(knn, Xtr, ytr, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
y_pred_train=knn.predict(Xtr)
print('KNN MSE CV - %.4f'%cv.mean(), 'KNN Train MSE - %.4f'%MSE(ytr, y_pred_train), 
      'KNN Test MSE - %.4f'%MSE(ytest, y_pred))
# #play with different k factor  and plot comparison
# a=[]
# b=[]
# for i in range(1, 40):
#     knn=KNeighborsClassifier(n_neighbors=i)
#     knn.fit(Xtr, ytr)
#     knn.predict(Xtest)
#     b.append(knn.score(Xtest, ytest))
#     a.append(knn.score(Xtr, ytr))
# plt.plot(a)
# plt.plot(b)
# plt.xlabel('K-number of neighbors')
# plt.ylabel('Accuracy')
# plt.legend(['train','test'])

# #GridSearch CV (cross-validation)
# from sklearn.model_selection import GridSearchCV
# param_grid = {'n_neighbors': np.arange(1, 50)}
# knn_cv = GridSearchCV(knn, param_grid, cv=5)
# knn_cv.fit(Xtr, ytr)
# print(knn_cv.best_params_, knn_cv.best_score_)

#%% (2) logistic regresion

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(C=0.33, max_iter=10000)
logreg.fit(Xtr, ytr)
y_pred = logreg.predict(Xtest)
print('LogReg - %.4f'%logreg.score(Xtest, ytest))

cv = - cross_val_score(logreg, Xtr, ytr, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
y_pred_train=logreg.predict(Xtr)
print('Logreg MSE CV - %.4f'%cv.mean(), 'logreg Train MSE - %.4f'%MSE(ytr, y_pred_train), 
      'logreg Test MSE - %.4f'%MSE(ytest, y_pred))
# Result around 79-81% accuracy

# #GridSearch CV (cross-validation)
# from sklearn.model_selection import GridSearchCV
# param_grid = {'C': np.logspace(-5, 5, 50)}
# logreg_cv = GridSearchCV(logreg, param_grid, cv=5)
# logreg_cv.fit(X, y)
# print(logreg_cv.best_params_, logreg_cv.best_score_)

#%% (3) Naive Bayes

from sklearn.naive_bayes import GaussianNB 
nb = GaussianNB(priors=(0.4,0.6), var_smoothing=0.0017)
nb.fit(Xtr, ytr)
y_pred = nb.predict(Xtest)
print('NaiveBayes - %.4f'%nb.score(Xtest, ytest))

cv = - cross_val_score(nb, Xtr, ytr, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
y_pred_train=nb.predict(Xtr)
print('nb MSE CV - %.4f'%cv.mean(), 'nb Train MSE - %.4f'%MSE(ytr, y_pred_train), 
      'nb Test MSE - %.4f'%MSE(ytest, y_pred))

# Result 77-79# accuracy

#%% (4) TREE
from sklearn.tree import DecisionTreeClassifier 
tree = DecisionTreeClassifier(criterion='entropy',max_depth=1, min_samples_leaf=5)
tree.fit(Xtr, ytr)
y_pred = tree.predict(Xtest)
print('Tree - %.4f'%tree.score(Xtest, ytest))

cv = - cross_val_score(tree, Xtr, ytr, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
y_pred_train=tree.predict(Xtr)
print('tree MSE CV - %.4f'%cv.mean(), 'tree Train MSE - %.4f'%MSE(ytr, y_pred_train), 
      'tree Test MSE - %.4f'%MSE(ytest, y_pred))

# Result 75%

#%% (5) SVM ???
from sklearn.svm import SVC 
svc = SVC()
svc.fit(Xtr, ytr)
y_pred = svc.predict(Xtest)
print('SVC - %.4f'%svc.score(Xtest, ytest))

cv = - cross_val_score(svc, Xtr, ytr, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
y_pred_train=svc.predict(Xtr)
print('svc MSE CV - %.4f'%cv.mean(), 'svc Train MSE - %.4f'%MSE(ytr, y_pred_train), 
      'svc Test MSE - %.4f'%MSE(ytest, y_pred))

#%% Stacking - Voting Classifier 
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

classifiers = [('knn',knn),('logreg',logreg),('tree',tree),('svc',svc)]
vc = VotingClassifier(estimators=classifiers)
vc.fit(Xtr, ytr)
y_pred = vc.predict(Xtest)
print('Voting classy %.4f'%accuracy_score(ytest, y_pred))


#%% ROC curve (can be done for all models)

# from sklearn.metrics import roc_curve 
# y_pred_prob= nb.predict_proba(Xtest)[:,1]
# fpr, tpr, tresholds = roc_curve(ytest, y_pred_prob)
# plt.plot(fpr, tpr)
# plt.show()

#%% plot everything

# x1 = train.groupby('Age').count()
# plt.bar(x1.index, height=x1['Survived'])
# plt.show()

# x2 = train.groupby('Pclass').count()
# plt.bar(x2.index, height=x2['Survived'])
# plt.show()

# x3 = train.groupby('Sex').count()
# plt.bar(x3.index, height=x3['Survived'])
# plt.show()

# x4 = train.groupby('SibSp').count()
# plt.bar(x4.index, height=x4['Survived'])
# plt.show()

# x5 = train.groupby('Parch').count()
# plt.bar(x5.index, height=x5['Survived'])
# plt.show()