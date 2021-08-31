# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 10:43:54 2021

@author: Admin
"""

from frozen_agent import *
from frozen_environment import *
#define 12x12 map
map = MAPS['12x12']
#Traverse all points on the map judge_repeat
for i in range(len(map)):
    for j in range(len(map[i])):
        #If the location on the map is 'F', select it as the starting point for training
        if map[i][j] == 'F':
            agent = DQNFrozenLakeAgent(start=[i,j])
            agent.run_training()
