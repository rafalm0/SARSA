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

    if not model.model_name:
        # Choose A from S
        action = model.choose(env, state, greedy)

    while not ended:

        if model.model_name:
            # Choose A from S
            action = model.choose(env, state, greedy)

        # take A from S and get S'
        new_state, reward, ended, time_limit, prob = env.step(action)

        if model.model_name:
            if greedy:  # testing episode wont update
                # updating
                model.update(reward, state, action, new_state, None)
        else:
            # choose A' from S'
            new_action = model.choose(env, new_state, greedy)

            if greedy:  # testing episode wont update
                # updating
                model.update(reward, state, action, new_state, new_action)
            # A <- A'
            action = new_action

        # S <- S'
        state = new_state

        if time_limit:
            break

    return {'reward': reward, 'mode': greedy}


# In[5]:


# defining process for each of the segments
def segment(model, env, training, verbose):
    results = {}

    for i, mode in enumerate(training):
        if verbose:
            print(f"-{i + 1}", end='')
        episode_result = episode(model, env, mode)
        results[i + 1] = episode_result

    return results


# In[6]:


# defining process for each of the runs
def run(model, env, segments_n=500, training=np.append(np.zeros(10), [1]), verbose=True):
    run_results = {}
    for i, mode in enumerate(range(segments_n)):
        if verbose:
            print(f"\n{i + 1}th Segment:", end='')

        run_results[i + 1] = segment(model, env, training, verbose)

    return run_results


# <div style="border-bottom: 3px solid black; margin-bottom:5px"></div>
# <div style="border-bottom: 3px solid black; margin-bottom:5px"></div>
# 
# 
# ## Running the model

# In[7]:


# configurations

temperatures = [.0,.1,.01]
learning_rates = [.85, .5, .15]
expected = ['expected', 'classic']
n_runs = 10

# In[8]:


# Declaring the model

models = []
general_results = {}
for type_ in expected:
    general_results[type_] = {}
    for alpha in learning_rates:
        general_results[type_][alpha] = {}
        for temp in temperatures:
            general_results[type_][alpha][temp] = {}

            bool_type = True if type_ == 'expected' else False
            models.append(sarsa(matrix.copy(), alpha=alpha, temperature=temp, expected=bool_type))

# In[9]:


# Runing the training

for model in models:
    model_type = 'expected' if model.expected else 'classic'
    print(f'Training on |temperature: {str(model.temp)}\t| alpha: {str(model.a)} \t| {model_type} Sarsa')

    for i in range(n_runs):
        general_results[model_type][model.a][model.temp][i + 1] = run(model, env, verbose=False)

# In[10]:


df = pd.DataFrame.from_dict({(a, b, c, d, e, f): general_results[a][b][c][d][e][f]
                             for a in general_results.keys()
                             for b in general_results[a].keys()
                             for c in general_results[a][b].keys()
                             for d in general_results[a][b][c].keys()
                             for e in general_results[a][b][c][d].keys()
                             for f in general_results[a][b][c][d][e].keys()},
                            orient='index')

# In[11]:


df = df.reset_index()
df = df.rename(
    columns={'level_0': 'sarsa', 'level_1': 'alpha', 'level_2': 'temperature', 'level_3': 'run', 'level_4': 'segment',
             'level_5': 'episode'})

# <div style="border-bottom: 3px solid black; margin-bottom:5px"></div>
# <div style="border-bottom: 3px solid black; margin-bottom:5px"></div>
# 
# ## Plotting
# Pick 3 settings of the temperature parameter used in the exploration and 3 settings of the learning rate. You need to plot:

# ### First question

# One u-shaped graph that shows the effect of the parameters on the final training performance,
# expressed as the return of the agent (averaged over the last 10 training episodes and the 10
# runs); note that this will typically end up as an upside-down u.

# In[12]:


df_training = df[df['mode'] == 0]

for model in df_training['sarsa'].unique():
    df_ = df_training[df_training['sarsa'] == model]
    df_ = df_.apply(lambda a: a, axis=1)
    df_['seg-ep'] = df_['segment'] * max(df_['episode']) + df_['episode'] - 10
    df_ = df_[['alpha', 'temperature', 'run', 'reward', 'seg-ep']]
    df_ = df_.groupby(['alpha', 'temperature', 'seg-ep']).mean()
    df_ = df_.dropna().reset_index()
    df_ = df_.groupby(['alpha', 'temperature']).rolling(10).mean()['reward'].dropna()
    print(df_)

# In[ ]:
