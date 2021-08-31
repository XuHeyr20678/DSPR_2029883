# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 05:58:15 2021

@author: Admin
"""

from imodels import CLASSIFIERS,REGRESSORS
import numpy as np
import pandas as pd
import os
import time
from IPython.display import clear_output
from sklearn.metrics import accuracy_score


#使用classifier做可解释AI
regressor = REGRESSORS[0]
model = regressor()
           
data = pd.read_csv("../Data/input/boston_1.csv", index_col=0)
y_train = data.medv.values
x_train = data.drop("medv", axis=1)
features = x_train.columns
x_train = x_train.values




data_test = pd.read_csv("../Data/input/boston_test.csv", index_col=0)
y_test = data_test.medv.values
x_test = data_test.drop("medv", axis=1)
features = x_test.columns
x_test = x_test.values



model.fit(x_train,y_train,feature_names=features)
#Strategies for model prediction
rules = model.get_rules(exclude_zero_coef = False)
rules = rules[rules.coef != 0].sort_values("support", ascending=False)
print(rules)
rules.to_csv("Boston_rule.csv")
y_pre = model.predict(x_test)
y_pred = (y_pre.round(0)).astype(np.int)
print(y_pre)
print(y_pred)
print(y_test)
print(accuracy_score(y_test, y_pre))
print(model.visualize())