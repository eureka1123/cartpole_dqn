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
BATCH_SIZE = 40
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

def run_cartpole_random():
    env = gym.make(ENV_NAME)
    for i in range(20):
        state = env.reset()
        for t in range(100):
            env.render()
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()

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
    if os.path.exists(weights_path) and os.path.exists(states_path):
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
            state = np.reshape(state, [1, observation_size])
            step = 0
            while not done:
                step +=1
                if display:
                    env.render()
                action = return_action(dqn, state)
                next_state, reward, done, info = env.step(action)
                states.append(list(next_state))
                next_state = np.reshape(next_state, [1, observation_size])
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
    # print(max([i[0] for i in states]),max([i[1] for i in states]),max([i[2] for i in states]),max([i[3] for i in states]))
    max_state = ""
    max_num = 0
    min_state = ""
    min_num= len(states)
    state_dict = {}
    for state in states:
        tstate = tuple([round(i,2) for i in state])
        state_dict.setdefault(tstate,0)
        state_dict[tstate]+=1
        if state_dict[tstate]>max_num:
            max_num = state_dict[tstate]
            max_state = tstate
        if state_dict[tstate]<min_num:
            min_num = state_dict[tstate]
            min_state = tstate
    state_reshape = np.reshape(list(max_state), [1, observation_size])
    maxstate_tensor = Variable(torch.from_numpy(state_reshape)).float()
    state_reshape = np.reshape(list(min_state), [1, observation_size])
    minstate_tensor = Variable(torch.from_numpy(state_reshape)).float()
    print(max_state,max_num, dqn(maxstate_tensor), min_state,min_num,dqn(minstate_tensor))
    # sample_states = random.sample(states, 10)
    # action_dict = {}
    # for state in sample_states:
    #     state = env.reset()
    #     state_reshape = np.reshape(state, [1, observation_size])
    #     state_tensor = Variable(torch.from_numpy(state_reshape)).float()
    #     q_values = dqn(state_tensor)
    #     action = torch.argmax(q_values).item()
    #     action_dict.setdefault(action,[])
    #     action_dict[action].append(state)

    # print(action_dict)

            

class DQN(nn.Module):
    def __init__(self, observation_size, action_size):
        super(DQN, self).__init__()
        self.exploration_rate = EXPLORATION_MAX
        self.action_space = action_size

        self.fc1 = nn.Linear(observation_size, 24)
        self.fc2 = nn.Linear(24,24)
        self.fc3 = nn.Linear(24, action_size) 
        self.memory = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def learn(dqn, optimizer, criterion, state, action, reward, next_state, done):
    dqn.memory.append((state, action, reward, next_state, done))
    if len(dqn.memory) < BATCH_SIZE:
        return 
    batch = random.sample(dqn.memory, BATCH_SIZE)
    for state, action, reward, next_state, done in batch:
        state_tensor = Variable(torch.from_numpy(state)).float()
        next_state_tensor = Variable(torch.from_numpy(next_state)).float()
        new_q_value = reward
        if not done:
            new_q_value = reward + GAMMA * torch.max(dqn(next_state_tensor).detach())
        output = dqn(state_tensor)
        q_values = output.clone().detach()
        q_values[0][action] = new_q_value
        loss = criterion(output, q_values)
        loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    dqn.exploration_rate *= EXPLORATION_DECAY
    dqn.exploration_rate = max(EXPLORATION_MIN, dqn.exploration_rate)


def return_action(dqn, state):
    if np.random.rand() < dqn.exploration_rate:
        return random.randrange(dqn.action_space)
    state_tensor = Variable(torch.from_numpy(state)).float()
    q_values = dqn(state_tensor)
    return torch.argmax(q_values).item()

run_cartpole_dqn(300)
