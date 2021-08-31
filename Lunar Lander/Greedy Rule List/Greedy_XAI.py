# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 08:33:30 2021

@author: Admin
"""

from imodels import CLASSIFIERS,REGRESSORS
import numpy as np
import os
import time
from IPython.display import clear_output
from sklearn.metrics import accuracy_score


#使用classifier做可解释AI
classifier = CLASSIFIERS[1]
model = classifier()

def transform_state(state):
    return np.array(state)

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
