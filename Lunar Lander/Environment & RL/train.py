import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
from collections import deque
import random
import csv

GAMMA = 0.99
target_update_rate = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
headers = ['pos x','pos y','vel x','vel y','angle','angleVel','contact_1','contact_2','action']

class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, element):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = element
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return list(zip(*random.sample(self.memory, batch_size)))

    def __len__(self):
        return len(self.memory)


def transform_state(state):
    return np.array(state)


class DQN:
    def __init__(self, state_dim, action_dim, batch_size, capacity, eps):
        self.gamma = GAMMA
        self.eps = eps
        self.batch_size = batch_size
        self.memory = Memory(capacity)
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.build_model()

    def build_model(self):
        def init_weights(layer):
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)

        self.model = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim))

        self.model.apply(init_weights)

        self.target_model = copy.deepcopy(self.model)
        self.model.to(device)
        self.target_model.to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00003)


    def act(self, state, target=False):
        return self.model(torch.tensor(state).to(device).float()).max(0)[1].view(1, 1).item()

    def save(self):
        torch.save(self.model, "agent.pkl")

    def fit(self, batch):
        target_model = self.target_model
        state, action, reward, next_state, done = batch

        state = torch.tensor(state).to(device).float()
        next_state = torch.tensor(next_state).to(device).float()
        reward = torch.tensor(reward).to(device).float()
        action = torch.tensor(action).to(device)
        done = torch.tensor(done).to(device)

        target_q = torch.zeros(reward.size()[0]).float().to(device)
        with torch.no_grad():
            target_q[~done] = target_model(next_state).max(1)[0].detach()[~done]

        target_q = reward + target_q * self.gamma

        loss = F.smooth_l1_loss(self.model(state).gather(1, action.unsqueeze(1)), target_q.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()

        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    np.random.seed(9)
    env.seed(9)
    dqn = DQN(state_dim=8, action_dim=4, batch_size = 64, capacity = 100000, eps = 1 )
    episodes = 1000
    eps = dqn.eps
    total_steps = 0
    old_reward = -2000
    reward_list = deque(maxlen = 50)
    train_data = []

    for i in range(episodes):
        # env.render()
        train_data.append([])
        state = transform_state(env.reset())
        total_reward = 0
        steps = 0
        done = False
        while not done:

            if random.random() < eps:
                action = env.action_space.sample()
            else:
                action = dqn.act(state)
            next_state, reward, done, _ = env.step(action)
            # env.render()
            next_state = transform_state(next_state)
            total_reward += reward
            steps += 1
            total_steps += 1
            dqn.memory.push((state, action, reward, next_state, done))
            train = state.flatten()
            train = np.append(train ,action)
            train_data[-1].append(train)
            state = next_state
            if steps >= dqn.batch_size:
                dqn.fit(dqn.memory.sample(dqn.batch_size))

            if total_steps % target_update_rate == 0:
                dqn.target_model = copy.deepcopy(dqn.model)
        eps = max(eps*0.95, 0.001)
        if len(reward_list) > 0:
            if total_reward > np.max(reward_list) and total_reward > 250:
                dqn.save()
                np.save(f"lunar_{i}", train_data[-1])
                with open(f'lunar_{i}.csv', 'w')as f:
                    f_csv = csv.writer(f)
                    f_csv.writerow(headers)
                    f_csv.writerows(train_data[-1])
                if (i + 1) % 10 != 0:

                         print(f"episode: {i + 1},current reward: {total_reward}")
        reward_list.append(total_reward)

        if (i + 1) % 10 == 0:

            print(f"episode: {i + 1}, current reward: {total_reward}")

