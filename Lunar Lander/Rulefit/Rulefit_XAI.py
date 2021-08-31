# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 08:43:07 2021

@author: Admin
"""

from imodels import CLASSIFIERS,REGRESSORS
import numpy as np
import os
import time
from IPython.display import clear_output
from sklearn.metrics import accuracy_score


#使用classifier做可解释AI
regressor = REGRESSORS[0]
model = regressor()

x_train = []
y_train = []
#加载强化学习学好的策略作为数据集
for i in range(900):
    if os.path.exists(f'../Data/npy/lunar_{i+1}.npy'):
        policy = np.load(f'../Data/npy/lunar_{i+1}.npy',allow_pickle=True)
        for j in policy:
            x_train.append(np.array(j[0:8]))
            y_train.append(j[-1])


x_train = np.array(x_train)
y_train = np.array(y_train)

feature_names = ['x_axis','y_axis']


x_test = []
y_test = []
for i in range(100):
    if os.path.exists(f'../Data/npy/lunar_{i+900}.npy'):
        policy = np.load(f'../Data/npy/lunar_{i+900}.npy',allow_pickle=True) 
        for j in policy:
            x_test.append(np.array(j[0:8]))
            y_test.append(j[-1])

x_test = np.array(x_test)
y_test = np.array(y_test)


model.fit(x_train,y_train)
#Strategies for model prediction
rules = model.get_rules(exclude_zero_coef = False)
rules = rules[rules.coef != 0].sort_values("support", ascending=False)
print(rules)
rules.to_csv("rule.csv")
y_pre = model.predict(x_test)
y_pred = (y_pre.round(0)).astype(np.int)
print(y_pre)
print(y_pred)
print(y_test)
print(accuracy_score(y_test, y_pred))
print(model.visualize())