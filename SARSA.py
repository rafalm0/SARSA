#!/usr/bin/env python
# coding: utf-8

# # Assignment 2 - SARSA

# <div style="border-bottom: 3px solid black; margin-bottom:5px"></div>
# <div style="border-bottom: 3px solid black; margin-bottom:5px"></div>
# 
# 
# ## Importing packages

# In[1]:


import gymnasium as gym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
rnd = np.random.default_rng(112233)


# <div style="border-bottom: 3px solid black; margin-bottom:5px"></div>
# <div style="border-bottom: 3px solid black; margin-bottom:5px"></div>
# 
# ## Building Frozen lake 

# In[2]:


env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode='ansi')

matrix = np.zeros((env.observation_space.n, env.action_space.n))


# <div style="border-bottom: 3px solid black; margin-bottom:5px"></div>
# 
# ## Building Sarsa class

# In[3]:


class sarsa():
    def __init__(self, decision_matrix, alpha=.85, gamma=.95, temperature=.05, expected=False):
        self.a = alpha
        self.g = gamma
        self.q = decision_matrix
        self.temp = temperature
        self.expected = expected

        return

    def update(self, reward, state, action, next_state, next_action=None):  # next action can be none in the expected

        if self.expected: 
            self.q[state, action] = self.q[state, action] + self.a * (
                    reward + self.g * np.sum(self.q[next_state, :] * self.boltzmann(next_state))
                    - self.q[state, action])
        else:
            self.q[state, action] = self.q[state, action] + self.a * (
                    reward + self.g * self.q[next_state, next_action] - self.q[state, action])

        return None

    def choose(self, env, state, greedy):

        if np.max(self.q[state]) == 0:
            # random sampling
            chosen = rnd.choice(list(range(env.action_space.n)))
        elif greedy or (self.temp <= 0):  # temp 0 means greedy, and cannot go to boltzmann to avoid division by 0
            # greedy choice
            chosen = np.argmax(self.q[state])
        else:
            # boltzmann probability
            prob = self.boltzmann(state)
            chosen = rnd.choice(list(range(env.action_space.n)), p=prob)

        return chosen

    def boltzmann(self, state):
        actions = np.divide(self.q[state], self.temp)
        upper = np.exp(actions)
        lower = np.sum(upper)
        return upper / lower

        


# <div style="border-bottom: 3px solid black; margin-bottom:5px"></div>
# 
# ## Building the training process

# In[4]:


# defining one episode
def episode(model, env, greedy=0):
    env.reset()
    state = 0  # initializing the state
    ended = False
    reward = 0

    if not model.expected:
        # Choose A from S
        action = model.choose(env, state, greedy)

    while not ended:

        if model.expected:
            # Choose A from S
            action = model.choose(env, state, greedy)

        # take A from S and get S'
        new_state, reward, ended, time_limit, prob = env.step(action)

        if model.expected:
            if greedy: # testing episode wont update
                # updating
                model.update(reward, state, action, new_state, None)
        else:
            # choose A' from S'
            new_action = model.choose(env, new_state, greedy)
            
            if greedy: # testing episode wont update
                # updating
                model.update(reward, state, action, new_state, new_action)
            # A <- A'
            action = new_action

        # S <- S'
        state = new_state

        if time_limit:
            break

    return reward


# In[5]:


# defining process for each of the segments
def segment(model, env, training,verbose):
    train_results = []
    test_results = []

    for i, mode in enumerate(training):
        if verbose:
            print(f"-{i + 1}", end='')
        episode_result = episode(model, env, mode)
        if mode:
            train_results.append(episode_result)
        else:
            test_results.append(episode_result)

    return train_results, test_results


# In[6]:


# defining process for each of the runs
def run(model, env, segments_n=500, training=np.append(np.zeros(10), [-1]),verbose=True):
    run_results = []
    for i, mode in enumerate(range(segments_n)):
        if verbose:
            print(f"\n{i + 1}th Segment:", end='')
        train_results, test_results = segment(model, env, training,verbose)
        run_results.append([train_results, test_results])

    return np.array(run_results)


# <div style="border-bottom: 3px solid black; margin-bottom:5px"></div>
# <div style="border-bottom: 3px solid black; margin-bottom:5px"></div>
# 
# 
# ## Running the model

# In[7]:


# configurations

temperatures = [.5, .1, .001]
learning_rates = [.85, .5, .15]


# In[8]:


# Declaring the model

models = []
for temp in temperatures:
    for alpha in learning_rates:
        models.append(sarsa(matrix,alpha=alpha,temperature=temp, expected=True))
        


# In[9]:


# Runing the training
results = []
for model in models:
    results.append(run(model,env,verbose=False))


# In[ ]:


results


# In[ ]:


# plt.figure(figsize=(12, 5))
# plt.xlabel("Episode number")
# plt.ylabel("Outcome")
# ax = plt.gca()
# ax.set_facecolor('#efeeea')
# plt.bar(range(len(general_result)), general_result, color="#0A047A", width=1.0)
# plt.show()


# In[ ]:




