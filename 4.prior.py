#!/usr/bin/env python
# coding: utf-8

#todo MakeNote1 -> I stacked 3 frames and reshaped the PyTorch wrapper to have a shape (3,1,84,84)
#todo MakeNote2 -> The state and env.observation.shape don't have the same shape

#todo Check this code with original once again


# In[1]:


import math, random
import cv2
import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
import matplotlib.pyplot as plt


# <h3>Use Cuda</h3>

# In[3]:


USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


# <h2>Prioritized Replay Buffer</h2>

# <p>Prioritized Experience Replay: https://arxiv.org/abs/1511.05952</p>

# In[28]:


class NaivePrioritizedBuffer(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity   = capacity
        self.buffer     = []
        self.pos        = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    
    def push(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs  = prios ** self.prob_alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total    = len(self.buffer)
        weights  = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)
        
        batch       = list(zip(*samples))
        states      = np.concatenate(batch[0])
        actions     = batch[1]
        rewards     = batch[2]
        next_states = np.concatenate(batch[3])
        dones       = batch[4]
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)


# In[5]:


beta_start = 0.4
beta_frames = 1000 
beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)


# <h3>Synchronize current policy net and target net</h3>

# In[12]:


def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


# In[13]:




# <h2>Computing Temporal Difference Loss</h2>

# In[14]:


def compute_td_loss(batch_size, beta):
    state, action, reward, next_state, done, indices, weights = replay_buffer.sample(batch_size, beta)
    state = state.reshape(batch_size * T, 1, state.shape[-2], state.shape[-1])
    next_state = next_state.reshape(batch_size * T, 1, state.shape[-2], state.shape[-1])
    state      = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action     = Variable(torch.LongTensor(action))
    reward     = Variable(torch.FloatTensor(reward))
    done       = Variable(torch.FloatTensor(done))
    weights    = Variable(torch.FloatTensor(weights))

    q_values      = current_model(state)
    next_q_values = target_model(next_state)

    q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value     = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    
    loss  = (q_value - expected_q_value.detach()).pow(2) * weights
    prios = loss + 1e-5
    loss  = loss.mean()
        
    optimizer.zero_grad()
    loss.backward()
    replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
    optimizer.step()
    
    return loss


# <h1>Atari Environment</h1>

# In[17]:


from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch


# In[18]:


env_id = "FreewayNoFrameskip-v4"
env    = make_atari(env_id)
env    = wrap_deepmind(env)
env    = wrap_pytorch(env)


# In[36]:

T = env.observation_space.shape[0]
print("The number of time steps = ", T)
class CnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CnnDQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        self.conv1 = nn.Conv2d(input_shape[1], 32, kernel_size=8, stride=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.relu3 = nn.ReLU()

        self.fc = nn.Sequential(
            nn.Linear(9408, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        n,c,h,w = x.shape
        x = x.reshape(n//T,T,c,h,w)
        x = x.view(x.size(0), -1)

        # todo MakeNote3 -> Changed the shape in x.view to get back 32 states
        x = self.fc(x)
        return x
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = Variable(torch.FloatTensor(np.float32(state)), volatile=True)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(env.action_space.n)
        return action


# In[37]:


current_model = CnnDQN(env.observation_space.shape, env.action_space.n)
target_model  = CnnDQN(env.observation_space.shape, env.action_space.n)

if USE_CUDA:
    current_model = current_model.cuda()
    target_model  = target_model.cuda()
    
optimizer = optim.Adam(current_model.parameters(), lr=0.0001)

replay_initial = 10000
replay_buffer  = NaivePrioritizedBuffer(100000)

update_target(current_model, target_model)


# <h3>Epsilon greedy exploration</h3>

# In[38]:


epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 30000

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)


# In[27]:

beta_start = 0.4
beta_frames = 100000
beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)


# In[41]:




# <h3>Training</h3>

# In[42]:

num_frames = 1400000
batch_size = 32
gamma      = 0.99

losses = []
reward_step = np.empty(shape = num_frames)

all_rewards = []
episode_reward = 0

state = np.expand_dims(env.reset(), axis = 1)

for frame_idx in range(1, num_frames + 1):
    print("Frame = ", frame_idx)
    epsilon = epsilon_by_frame(frame_idx)
    action = current_model.act(state, epsilon)
    
    next_state, reward, done, _ = env.step(action)

    next_state = np.expand_dims(next_state, axis = 1)


    replay_buffer.push(state, action, reward, next_state, done)
    
    state = next_state
    episode_reward += reward
    reward_step[frame_idx - 1] = reward

    if done:
        state = np.expand_dims(env.reset(), axis = 1)
        all_rewards.append(episode_reward)
        np.savetxt('pr_dqn.out', all_rewards, delimiter=',')
        episode_reward = 0
        
    if len(replay_buffer) > replay_initial:
        beta = beta_by_frame(frame_idx)
        loss = compute_td_loss(batch_size, beta)
        losses.append(loss.data)
        

        
    if frame_idx % 1000 == 0:
        update_target(current_model, target_model)

    if frame_idx % 100000 == 0:
        print("Frame Index = ", frame_idx)
        np.savetxt('pr_dqn_step.out', reward_step, delimiter=',')
        


# In[ ]:




