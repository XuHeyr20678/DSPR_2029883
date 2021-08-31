# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 09:11:48 2021

@author: Admin
"""

import numpy as np
import os
import time
from IPython.display import clear_output
from sklearn.metrics import accuracy_score
from CN2 import CN2
import pickle
import pandas as pd

cn2 = CN2()
train_start = time.time()
print("-----------------------------------------------------")
rules = cn2.fit('lunar_small_train.csv')
train_end = time.time()
print('Training time: ', train_end - train_start, ' s')
print('Rules:')
cn2.print_rules(rules)
print('------------------------------------------------------')

with open('../Data/output/train_rules', 'wb') as f:
    pickle.dump(rules, f)

# These two lines can be used to load a previously computed set of rules.
# with open('../Data/output/iris_rules', 'rb') as f:
#   rules = pickle.load(f)

rules_performance, accuracy = cn2.predict('lunar_small_test.csv', rules)
print('Accuracy: ', accuracy)
print('Testing performance:')
keys = []
vals = []
for data in rules_performance:
    val = []
    for k, v in data.items():
        keys.append(k)
        val.append(v)
    vals.append(val)

table = pd.DataFrame([v for v in vals], columns=list(dict.fromkeys(keys)))
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(table)
table.to_csv('../Data/output/lunar_performance_1.csv')