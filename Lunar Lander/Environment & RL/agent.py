import gym
import os
import pickle
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "/agent.pkl")
        self.model.to(device)

    def act(self, state):
        state = torch.tensor(state).to(device).unsqueeze(0).float()
        with torch.no_grad():
            return self.model(state).max(1)[1].view(1, 1).item()


    def reset(self):
        pass