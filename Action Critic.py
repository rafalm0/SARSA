import pandas as pd
import pygame
import matplotlib.pyplot as plt
import math
import random
import os
import numpy as np
import gymnasium as gym
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
rnd = np.random.default_rng(112233)

env = gym.make('CartPole-v1')
env.reset()


class qlearning:
    def __init__(self, env, alpha=.85, gamma=.95, epsilon=.1, bins=10):
        self.a = alpha
        self.g = gamma
        self.q = self.gen_table(env, bins)
        self.e = epsilon
        self.n_bins = bins

        # changing bounds into more compact values to speed up training (fewer bins needed for this accuracy):
        self.env_space = [[3, -3],
                          [6, -6],
                          [0.300, -0.300],
                          [5, -5]]

        return

    def gen_table(self, env, bins):
        action_dim = env.action_space.n

        table = np.random.uniform(low=-0.001, high=0.001, size=(bins, bins, bins, bins, action_dim))

        self.q = table
        return self.q

    def update(self, reward, state, action, next_state, final=False):
        a, b, c, d, e = self.get_s(state, action)
        a_, b_, c_, d_ = self.get_s(next_state)

        self.q[a][b][c][d][e] = self.q[a][b][c][d][e] + self.a * (
                reward + self.g * np.max(self.q[a_][b_][c_][d_]) - self.q[a][b][c][d][e])

        return None

    def choose(self, env, state):

        if rnd.random() < self.e:
            # random sampling
            chosen = rnd.choice(list(range(env.action_space.n)))
        else:
            # greedy choice
            table = self.q
            for miniState in self.get_s(state):
                table = table[miniState]

            chosen = np.argmax(table)
        return chosen

    def get_s(self, state, action=None):
        indexes = []
        for i, feature in enumerate(state):
            max_value = self.env_space[i][0]
            min_value = self.env_space[i][1]

            if (feature > max_value) or (feature < min_value):
                raise ValueError(
                    f"Feature out of bounds for feature{str(i)} on bins : {str(feature)}  |min : {str(min_value)} - "
                    f"max :{str(max_value)}|")
            window_size = (max_value - min_value) / self.n_bins
            bin_loc = (feature - min_value) // window_size
            indexes.append(int(bin_loc))

        if action is None:
            return indexes
        else:
            return indexes + [action]


class AC():
    def __init__(self, env, alpha=(.85, .85), gamma=.95, epsilon=.1):
        self.a = alpha
        self.g = gamma
        self.e = epsilon
        space = env.reset()[0]
        self.act = create_actor(len(space), env.action_space.n, alpha[0])
        self.crit = create_critic(len(space), env.action_space.n, alpha[1])

        return

    def update(self, reward, state, action, next_state, final=False):
        state = np.reshape(state, [1, 4])
        next_state = np.reshape(next_state, [1, 4])
        curr_pred = self.crit.predict(state, verbose=0)[0]
        next_pred = self.crit.predict(next_state, verbose=0)[0]

        if final:
            expected = np.zeros((1,1))
            expected[0][0] = reward
        else:
            expected = reward + self.g * next_pred

        advantage = np.zeros((1, 2))
        advantage[0][action] = expected - curr_pred

        self.crit.fit(state, expected, epochs=1, verbose=0)
        self.act.fit(state, advantage, epochs=1, verbose=0)

        return

    def choose(self, env, state):
        if rnd.random() < self.e:
            chosen = rnd.choice(list(range(env.action_space.n)))
        else:
            probs = self.act.predict(np.reshape(state, [1, 4]), batch_size=1, verbose=0).flatten()
            chosen = np.random.choice(env.action_space.n, 1, p=probs)[0]
        return chosen


def create_actor(input_size, output_size, alpha):
    actor_model = Sequential()
    actor_model.add(Dense(24, input_dim=input_size, activation='relu', kernel_initializer='he_uniform'))
    actor_model.add(Dense(output_size, activation='softmax', kernel_initializer='he_uniform'))
    actor_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=alpha))
    return actor_model


def create_critic(input_size, output_size, alpha):
    critic_model = Sequential()
    critic_model.add(Dense(24, input_dim=input_size, activation='relu', kernel_initializer='he_uniform'))
    critic_model.add(Dense(1, activation='linear', kernel_initializer='he_uniform'))
    critic_model.compile(loss="mse", optimizer=Adam(learning_rate=alpha))
    return critic_model


def episode(model, env, render=False, penalty=250):
    state = env.reset()[0]
    if render:
        env.render()
    ended = False
    ep_reward = 0

    while not ended:

        action = model.choose(env, state)

        # take A from S and get S'
        new_state, reward, ended, time_limit, prob = env.step(action)

        if ended:
            reward -= penalty

        model.update(reward, state, action, new_state, final=ended)

        # S <- S'
        state = new_state
        ep_reward += reward
        if time_limit:
            break

    if render:
        env.close()
    return ep_reward


def run(model, env, episode_n=1000, verbose=True,penalty=250):
    run_results = []
    for i, mode in enumerate(range(episode_n)):
        if verbose and (len(run_results) > 1):

            print(f"\n{i + 1}th Segment: {np.mean(run_results)} avg reward", end='')
        reward = episode(model, env,penalty=penalty)
        run_results.append(reward)

    return run_results


# # Q learning configurations
n_bins = 10

epsilons = [.1,.2,.5]
learning_rates = [1/4,1/8,1/16]

n_runs = 10
rolling_window = 10

training_size = 10
testing_size = 1
df = None

# Running the training
for alpha in learning_rates:
    for epsilon in epsilons:
        print(f'Training on |Epsilon: {str(epsilon)}\t| Alpha: {str(alpha)}')

        episode_results = []
        for i in range(n_runs):
            result_df = pd.DataFrame()
            # creating model copies for each run
            n_model = qlearning(env, alpha=alpha, epsilon=epsilon,bins=n_bins)
            result_df['ep_reward'] = run(n_model, env, verbose=False)
            result_df['alpha'] = alpha
            result_df['epsilon'] = epsilon
            result_df['run'] = i
            if df is None:
                df = result_df.copy()
            else:
                df = pd.concat([df, result_df])

df.to_csv('Qlearning.csv',index=False,sep=';',encoding='utf-8')








# AC configurations
n_bins = 10

epsilons = [.2]
learning_rates = [(.05, .15)]

n_runs = 2
rolling_window = 10

training_size = 10
testing_size = 1

models = []

AC_results = {}
for alpha in learning_rates:
    AC_results[alpha] = {}
    for epsilon in epsilons:
        AC_results[alpha][epsilon] = {}

        # creating model to use as standard for each run config
        models.append(AC(env, alpha=alpha, gamma=1, epsilon=epsilon))

# Running the training

for model in models:
    print(f'Training on |Epsilon: {str(model.e)}\t| Alpha: {str(model.a)}')

    for i in range(n_runs):
        # creating model copies for each run
        n_model = AC(env, alpha=model.a, epsilon=model.e)
        AC_results[model.a][model.e][i] = run(n_model, env, verbose=True, penalty=250)

pass
print()
