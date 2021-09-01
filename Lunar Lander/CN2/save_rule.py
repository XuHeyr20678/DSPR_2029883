# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 13:46:02 2021

@author: Admin
"""

import numpy as np
import pandas as pd
import collections
import time
import pickle
from sklearn.metrics import accuracy_score

with open('../Data/output/lunar_rules_MDLP', 'rb') as f:
     rules = pickle.load(f)
     
print(rules)
rule_list=pd.DataFrame(data=rules)

rule_list.to_csv('../Data/output/lunar_rules_MDLP.csv',encoding='utf-8')