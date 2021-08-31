# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 10:28:23 2021

@author: Admin
"""

import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from frozen_environment import FrozenLakeEnv
from IPython.display import clear_output
import csv

headers = ['episode','step','state','action','reward']

class DQNFrozenLakeAgent: #Define the agent 

    def __init__(self,start):
        self.env = FrozenLakeEnv(map_name='12x12',start = start)

        self.states = np.identity(144)
        self.x = tf.placeholder(shape=[1, 144], dtype=tf.float32) #state

        self.W = tf.Variable(tf.random_uniform([144, 4], 0, 0.1)) #Action

        self.Q = tf.matmul(self.x, self.W)
        self.Q_hat = tf.placeholder(shape=[1, 4], dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.Q_hat - self.Q))
        self.train = tf.train.GradientDescentOptimizer(learning_rate=0.1)\
            .minimize(self.loss)
        self.agent_name = f'{start[0]*12+start[1]+1}'
        self.gamma = 0.90
        self.epsilon = 1
        self.decay_rate = 0.001
        self.max_step = 1000

        self.num_episodes = 500_000
        self.avg_rewards = []

        init =tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init)
        self.avg_rewards = []

    def test_matrix(self, Q, episode):
        total_reward = 0
        for i in range(100):
            state = self.env.reset()
            done = False
            while not done:
                Q_ = self.session.run(Q, feed_dict={self.x: self.states[state:state + 1]})
                a = np.argmax(Q_, 1)[0]

                state, r, done, _ = self.env.step(a)
                total_reward += r

        result = total_reward / 100
        print('Episode: {:,}, Average reward: {}'.format(episode, result))
        return result

    def s_convert(self,state):
        #Create the row and col
        row = state//12
        col = state%12

        return [row + 1,col + 1]

    def epsilon_greedy(self, Q_pred,episode):
        """
        Returns the next action by exploration with probability epsilon and
        exploitation with probability 1-epsilon.
        """
        if np.random.random() <= self.epsilon and episode < 500:
            return self.env.action_space.sample()
        else:
            return np.argmax(Q_pred, 1)[0]

    def decay_epsilon(self, episode):
        """
        Decaying exploration with the number of episodes.
        """
        self.epsilon = 0.1 + 0.9 * np.exp(-self.decay_rate * episode)

    def run_training(self):
        """Training the agent to find the frisbee on the frozen lake"""

        self.avg_rewards = []
        self.episode_len = np.zeros(self.num_episodes)
        train_data = []

        for episode in range(self.num_episodes):

            state = self.env.reset()
            done = False
            step = 0
            train_data.append([])
            while not done and step < self.max_step:
                
                # Predicted Q

                Q_pred = self.session.run(self.Q, {self.x: self.states[state:state + 1]})
                action = self.epsilon_greedy(Q_pred,episode)
                new_state, reward, done, _ = self.env.step(action)

                # Actual Q after performing an action
                Q_true = reward + self.gamma * np.max(
                    self.session.run(self.Q, {self.x: self.states[new_state:new_state + 1]}))

                Q_pred[0, action] = Q_true

                # Calculate loss and train the agent
                self.session.run(self.train, feed_dict={self.x: self.states[state:state + 1], self.Q_hat: Q_pred})
                state_convert = self.s_convert(state)

                state = new_state

                self.episode_len[episode] += 1

                step += 1

                train_data[-1].append([episode, step, state_convert, action, reward])

            self.decay_epsilon(episode)
            print(f'episode = {episode}')
            #If the end is reached and the path is not repeated, record the track of the episode
            if reward == 1 and not self.judge_repeat(train_data[-1]):
                #write the data
                with open(f'dqn_{self.agent_name}.csv', 'w')as f:
                    f_csv = csv.writer(f)
                    f_csv.writerow(headers)
                    f_csv.writerows(train_data[-1])

                np.save(f'policy_{self.agent_name}', train_data[-1])
                print("You get the frisbee🥏")
                break

    def judge_repeat(self,train_data):
        state = []
        for i in train_data:
            state.append(str(i[2]))

        print(len(state),len(set(state)))
        if len(state) == len(set(state)):
           return False
        else:
            return True

    def plot(self):
        """Plot the episode length and average rewards per episode"""

        fig = plt.figure(figsize=(20, 5))

        episode_len = [i for i in self.episode_len if i != 0]

        rolling_len = pd.DataFrame(episode_len).rolling(100, min_periods=100)
        mean_len = rolling_len.mean()
        std_len = rolling_len.std()

        plt.plot(mean_len, color='red')
        plt.fill_between(x=std_len.index, y1=(mean_len - std_len)[0],
                         y2=(mean_len + std_len)[0], color='red', alpha=.2)

        plt.ylabel('Episode length')
        plt.xlabel('Episode')
        plt.title(
            f'Frozen Lake - Length of episodes (mean over window size 100)')
        fig.show()

        fig = plt.figure(figsize=(20, 5))

        plt.plot(self.avg_rewards, color='red')
        plt.gca().set_xticklabels(
            [i + i * 4999 for i in range(len(self.avg_rewards))])

        plt.ylabel('Average Reward')
        plt.xlabel('Episode')
        plt.title(f'Frozen Lake - Average rewards per episode ')
        fig.show()

def s_convert(state):
    row = state//12
    col = state%12

    return [row + 1,col + 1]

def play(agent, num_episodes=1):
    """Let the agent play Frozen Lake"""
    time.sleep(2)
    train_data = []
    for episode in range(num_episodes):
        state = agent.env.reset()
        done = False
        print('❄️🕳❄️ Frozen Lake - Episode ', episode + 1,
              '❄️🥏❄️ \n\n\n\n\n\n\n\n')

        time.sleep(1.5)

        steps = 0
        while not done:
            clear_output(wait=True)
            agent.env.render()
            time.sleep(0.3)

            Q_ = agent.session.run(agent.Q, feed_dict={agent.x: agent.states[state:state + 1]})
            action = np.argmax(Q_, 1)[0]
            state_convert = s_convert(state)
            state, reward, done, _ = agent.env.step(action)
            steps += 1

            train_data.append([episode, steps, state_convert, action, reward])

        np.save(f'policy_{agent.agent_name}',train_data)

        clear_output(wait=True)
        agent.env.render()

        if reward == 1:
            print(f'You have found your frisbee 🥏 in {steps} steps.')
            time.sleep(2)
        else:
            print('You fell through a hole 🕳, Game Over! Please try again!')
            time.sleep(2)
        with open('dqn_all.csv', 'w')as f:
            f_csv = csv.writer(f)
            f_csv.writerow(headers)
            f_csv.writerows(train_data)
        clear_output(wait=True)

if __name__ == '__main__':

    agent = DQNFrozenLakeAgent()
    agent.run_training()
    play(agent)
    agent.plot()