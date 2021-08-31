# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 10:11:58 2021

@author: Admin
"""

#This environment is built with reference to the source code in GYM.
#Because the environment files in GYM are difficult to find and modify,
#I use the environment I built to modify the map.
#import
import sys
import numpy as np
import random
from six import StringIO, b
from gym import utils
from gym.envs.toy_text import discrete
from contextlib import closing

#direction
L = 0;
D = 1;
R = 2;
U = 3;

#Map 
MAPS = {"12x12":[
        "FFFHFFFHFFFF",
        "FHFFFFFFFFFF",
        "FFHFFHFFFFFF",
        "FFFFFFFFFFFH",
        "FHFFFFHFFFFF",
        "FHFFFFFFFHFF",
        "FHFFFFHFFFFF",
        "FHFFHFFFFFHF",
        "FFFFFFFFHFFF",
        "FFFFFHFFFFFH",
        "FFFFFFFFHFFF",
        "FFFFFFHFFFFG"
     ],
     "16x16": [
        "FFFFFHFFFFFFHFFF",
        "FHFHFFFFFHFFFFFF",
        "FFFFHFFHFFFFFFFH",
        "HFFFFFHFFHFFFFFF",
        "FFFFHFFHFFFFFFFF",
        "FFFHFFFHFFFFFFHF",
        "FFFFFFFFFFHFFFHF",
        "FFFFFFFFHFFFHFFF",
        "FFFFFFFHFHFFFFFF",
        "FFFFHFFFFFFHFFFF",
        "FFHFHFFFFFFFFFFF",
        "FFFFHFFFFFHFHFFF",
        "FHFFFFFFHFFFFFFF",
        "HFFFFFFHFFFFFFFF",
        "FFFFFFFFFFFFHFFF",
        "FFFHFFFHFFHFFFFG"
    ]
}

#Define the original environment
class FrozenLakeEnv(discrete.DiscreteEnv):
    """
    Winter is here. You and your friends were tossing around a frisbee at the park
    when you made a wild throw that left the frisbee out in the middle of the lake.
    The water is mostly frozen, but there are a few holes where the ice has melted.
    If you step into one of those holes, you'll fall into the freezing water.
    At this time, there's an international frisbee shortage, so it's absolutely imperative that
    you navigate across the lake and retrieve the disc.
    However, the ice is slippery, so you won't always move in the direction you intend.
    The surface is described using a grid like the following
        SFFF
        FHFH
        FFFH
        HFFG
    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located
    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.
    """
    
    metadata = {'render.modes': ['human', 'ansi']}
    
    def __init__(self, map_name="4x4",is_slippery=False,start=None):
        #Define the Starting point (S)
        desc = np.copy(MAPS[map_name])
        desc = desc.tolist()
        desc[start[0]] = list(desc[start[0]])
        desc[start[0]][start[1]] = 'S'
        s = ''.join(desc[start[0]])
        desc[start[0]] = s
        print(desc)

        self.desc = desc = np.asarray(desc,dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        
        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row*ncol + col

        def inc(row, col, a):
            if a == L:
                col = max(col-1,0)
            elif a == D:
                row = min(row+1,nrow-1)
            elif a == R:
                col = min(col+1,ncol-1)
            elif a == U:
                row = max(row-1,0)
            return (row, col)
        #Move to the next state with the selected action
        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'GH':
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:  # add 0 to the list to add slippery
                                newrow, newcol = inc(row, col, b)
                                newstate = to_s(newrow, newcol)
                                newletter = desc[newrow, newcol]
                                done = bytes(newletter) in b'GH'
                                if newletter == b'G':
                                    rew = 1;
                                elif newletter == b'H':
                                    rew = -1;
                                else:
                                    rew = 0;
                                li.append((1.0 / 3.0, newstate, rew, done))
                        else:
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = bytes(newletter) in b'GH'
                            if newletter == b'G':
                                rew = 1;
                            elif newletter == b'H':
                                rew = -1;
                            else:
                                rew = 0;
                            li.append((1.0, newstate, rew, done))
                            
        super(FrozenLakeEnv, self).__init__(nS, nA, P, isd)
        
    def render(self, mode='human'):
        # Map Visualization
        map_em = {'S': '🏃', 'F': '❄', 'H': '🕳', 'G': '🥏'}

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[map_em[c.decode('utf-8')] for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left","Down","Right","Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
