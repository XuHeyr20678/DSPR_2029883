# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 08:06:44 2021

@author: Admin
"""

from imodels import CLASSIFIERS
import numpy as np
import os
import time
import pandas as pd
from IPython.display import clear_output
from sklearn.metrics import accuracy_score

def convert(state):
    # Converts ordinals to rows and columns
    row = state // 12
    col = state % 12

    return [row + 1, col + 1]

#Use classifier to be XAI AI algorithm
classifier = CLASSIFIERS[1]
model = classifier()

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
print(model)
Rulelist =model.print_list()
print(Rulelist)
y_pre = model.predict(x_test)
print(y_pre)
print(y_test)
print(accuracy_score(y_test, y_pre))

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
                #使用classifier预测动作，结果为浮点型，四舍五入取整
                print(([convert(state)])[0])
                print(model.predict([convert(state)])[0])
                action = round(model.predict([convert(state)])[0])
                action = int(action)

                #使用生成的动作在环境中执行
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