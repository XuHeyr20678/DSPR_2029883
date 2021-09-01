# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 04:33:44 2021

@author: Admin
"""

from imodels import CLASSIFIERS,REGRESSORS
import numpy as np
import pandas as pd
import os
import time
from IPython.display import clear_output
from sklearn.metrics import accuracy_score


def transform_state(state):
    return np.array(state)

#使用classifier做可解释AI
regressor = REGRESSORS[0]
model = regressor()
           
data = pd.read_csv("../Data/input/train_original.csv")
y_train = data.action.values
x_train = data.drop("action", axis=1)
features = x_train.columns
x_train = x_train.values




data_test = pd.read_csv("../Data/input/test_original.csv")
y_test = data_test.action.values
x_test = data_test.drop("action", axis=1)
features = x_test.columns
x_test = x_test.values



model.fit(x_train,y_train,feature_names=features)
#Strategies for model prediction
rules = model.get_rules(exclude_zero_coef = False)
rules = rules[rules.coef != 0].sort_values("support", ascending=False)
print(rules)
rules.to_csv("Rulefit_Rule.csv")
y_pre = model.predict(x_test)
y_pred = (y_pre.round(0)).astype(np.int)
print(y_pre)
print(y_pred)
print(y_test)
print(accuracy_score(y_test, y_pred))
print(model.visualize())

import gym
env = gym.make("LunarLander-v2")
np.random.seed(919)
env.seed(919)
total = 1000
success = 0

for i in range(100):
    # env.render()
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = np.round(model.predict([state]))
        action = int(action[0])
        if action < 0:
            action = 1
        next_state, reward, done, _ = env.step(action)
        # env.render()
        next_state = transform_state(next_state)
        total_reward += reward
        state = next_state
    print(f'total reward : {total_reward}')
    if total_reward > 200:
        success += 1
print(f'success rate = {success/total}')