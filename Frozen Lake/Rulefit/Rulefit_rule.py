# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 08:16:29 2021

@author: Admin
"""

from imodels import REGRESSORS
import numpy as np
import os
import time
from IPython.display import clear_output
from sklearn.metrics import accuracy_score
import pandas as pd

def convert(state):
    # Converts ordinals to rows and columns
    row = state // 12
    col = state % 12

    return [row + 1, col + 1]

#Use classifier to be XAI AI algorithm
regressor = REGRESSORS[0]
model = regressor()


data = pd.read_csv("../Data/input/original_train.csv")
y_train = data.action.values
x_train = data.drop("action", axis=1)
features = x_train.columns
x_train = x_train.values




data_test = pd.read_csv("../Data/input/original_test.csv")
y_test = data_test.action.values
x_test = data_test.drop("action", axis=1)
features = x_test.columns
x_test = x_test.values



model.fit(x_train,y_train, feature_names = features)
#Strategies for model prediction
rules = model.get_rules(exclude_zero_coef = False)
rules = rules[rules.coef != 0].sort_values("support", ascending=False)
print(rules)
rules.to_csv("Rulefit_rule.csv")
y_pre = model.predict(x_test)
y_pred = (y_pre.round(0)).astype(np.int)
print(y_pre)
print(y_pred)
print(y_test)
print(accuracy_score(y_test, y_pred))
print(model.visualize())

from frozen_environment import FrozenLakeEnv,MAPS
#load environment
map = MAPS['12x12']
total_count = 123
success = 0
fail = 0
# Traverse judge_repeat for all points on the map
for i in range(len(map)):
    for j in range(len(map[i])):
        #If the location on the map is 'FROZEN', select it as the starting point for training
        if map[i][j] == 'F':
            env = FrozenLakeEnv(map_name='12x12',start=[i,j])
            time.sleep(2)

            state = env.reset()
            done = False


            time.sleep(1.5)

            steps = 0

            while not done:
                clear_output(wait=True)
                env.render()
                time.sleep(0.3)
                #Use classifier to predict actions, results are floating point, rounded to the nearest whole number
                print(([convert(state)])[0])
                print(model.predict([convert(state)])[0])
                action = round(model.predict([convert(state)])[0])
                action = int(action)

                #Execute in the environment using the generated actions
                state, reward, done, _ = env.step(action)
                steps += 1

            clear_output(wait=True)
            env.render()

            if reward == 1:
                print(f'You have found your frisbee 🥏 in {steps} steps.')
                time.sleep(2)
                success += 1
            else:
                print('You fell through a hole 🕳, Game Over! Please try again!')
                time.sleep(2)
                fail += 1
            clear_output(wait=True)

print(total_count,success,fail)
print(f'success rate = {success/total_count}')
