#inspired by https://github.com/gsurma/cartpole
import random
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt 
import os
import csv

GAMMA = .95
ENV_NAME = "CartPole-v1"
LEARNING_RATE = .01
BATCH_SIZE = 30
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
BoolTensor = torch.cuda.BoolTensor if use_cuda else torch.BoolTensor


def run_cartpole_dqn(threshold_step = 250):
    weights_path = "model_weights"
    states_path = "states.csv"
    env = gym.make(ENV_NAME)
    observation_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    dqn = DQN(observation_size, action_size)

    optimizer = optim.Adam(dqn.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    run = 0
    step = 0
    display = False
    states = []
    if False and os.path.exists(weights_path) and os.path.exists(states_path):
        dqn.load_state_dict(torch.load(weights_path))
        with open (states_path, "r") as f:
            reader = csv.reader(f)
            states = list(list(i) for i in reader)
        states = [[float(i) for i in j] for j in states]
    
    else:
        while not display:
            if step >= threshold_step:
                display = True
            done = False
            env = gym.make(ENV_NAME)
            run += 1
            state = env.reset()
            # state = np.reshape(state, [1, observation_size])
            step = 0
            while not done:
                step +=1
                if display:
                    env.render()
                action = return_action(dqn, state)
                next_state, reward, done, info = env.step(action)
                # states.append(list(next_state))
                # next_state = np.reshape(next_state, [1, observation_size])
                if done:
                    reward = -reward
                learn(dqn, optimizer, criterion, state, action, reward, next_state, done)

                state = next_state
                if done:
                    print("run: ", run, " score: ", step)
                    env.close()

        torch.save(dqn.state_dict(), weights_path)
        # print(states)
        print("num states", len(states))
        with open(states_path, "w") as f:
            writer = csv.writer(f)
            writer.writerows(states)

class DQN(nn.Module):
    def __init__(self, observation_size, action_size):
        super(DQN, self).__init__()
        self.exploration_rate = EXPLORATION_MAX
        self.action_space = action_size

        self.fc1 = nn.Linear(observation_size, 50)
        self.fc2 = nn.Linear(50,50)
        self.fc3 = nn.Linear(50, action_size) 
        self.memory = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def learn(dqn, optimizer, criterion, state, action, reward, next_state, done):
    dqn.train()
    dqn.memory.append((FloatTensor([state]), LongTensor([[action]]), FloatTensor([reward]), FloatTensor([next_state]), FloatTensor([0 if done else 1])))

    if len(dqn.memory) < BATCH_SIZE:
        return 
    batch = random.sample(dqn.memory, BATCH_SIZE)
    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*batch)

    batch_state  = Variable(torch.cat(batch_state))
    batch_action = Variable(torch.cat(batch_action))
    batch_reward = Variable(torch.cat(batch_reward))
    batch_next_state = Variable(torch.cat(batch_next_state))
    batch_done = Variable(torch.cat(batch_done))

    current_q_values = dqn(batch_state).gather(1, batch_action).view(BATCH_SIZE)
    max_next_q_values = dqn(batch_next_state).detach().max(1)[0]
    expected_q_values = ((GAMMA * max_next_q_values)*batch_done + batch_reward)
    loss = criterion(current_q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    
    optimizer.step()
    dqn.exploration_rate *= EXPLORATION_DECAY
    dqn.exploration_rate = max(EXPLORATION_MIN, dqn.exploration_rate)


def return_action(dqn, state):
    if np.random.rand() < dqn.exploration_rate:
        return random.randrange(dqn.action_space)
    state_tensor = Variable(FloatTensor([state]))
    q_values = dqn(state_tensor)
    return torch.argmax(q_values).item()

run_cartpole_dqn(500)

