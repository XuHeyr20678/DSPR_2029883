# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 08:44:36 2021

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
classifier = CLASSIFIERS[1]
model = classifier()

def transform_state(state):
    return np.array(state)

data = pd.read_csv("../Data/input/train_1.csv")
y_train = data.action.values
x_train = data.drop("action", axis=1)
features = x_train.columns
x_train = x_train.values




data_test = pd.read_csv("../Data/input/test_1.csv")
y_test = data_test.action.values
x_test = data_test.drop("action", axis=1)
features = x_test.columns
x_test = x_test.values


model.fit(x_train,y_train,feature_names=features)
#Strategies for model prediction
print(model)
Rulelist =model.print_list()
print(Rulelist)
y_pre = model.predict(x_test)
print(y_pre)
print(y_test)
print(accuracy_score(y_test, y_pre))

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