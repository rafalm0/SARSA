import os
import gymnasium as gym
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
rnd = np.random.default_rng(112233)

env = gym.make('CartPole-v1')
env.reset()


num_states = 10 ** len(env.reset()[0])
num_actions = env.action_space.n
policy = np.ones((num_states, num_actions)) / num_actions
V = np.zeros(num_states) # value function

# Set alpha  and gamma
alpha_critic = 0.1
alpha_actor = 0.01
g = 0.99
e = .2


# tabular state value function
def get_s(s, n_bins=10):
    env_space = [[3, -3],
                 [6, -6],
                 [0.300, -0.300],
                 [5, -5]]
    indexes = 0
    for i, feature in enumerate(s):
        max_value = env_space[i][0]
        min_value = env_space[i][1]

        if (feature > max_value) or (feature < min_value):
            raise ValueError(
                f"Feature out of bounds for feature{str(i)} on bins : {str(feature)}  |min : {str(min_value)} - "
                f"max :{str(max_value)}|")
        window_size = (max_value - min_value) / n_bins
        bin_loc = (feature - min_value) // window_size
        indexes += int(bin_loc) * (10 ** i)

    return indexes


# Run the main loop
rewards_history = [0, 0, 0, 0, 0, 0, 0, 0, 0]
for episode in range(1000):
    state = get_s(env.reset()[0])
    done = False

    # eligibility traces
    z_critic = np.zeros(num_states)
    z_actor = np.zeros((num_states, num_actions))

    history = []

    while not done:

        action_probs = np.exp(policy[state]) / np.sum(np.exp(policy[state]))

        # e-greedy
        if rnd.random() < e:
            action = rnd.choice(list(range(num_actions)))  # random action to increase exploration
        else:
            action = np.random.choice(num_actions, p=action_probs)  # softmax action

        # Take action A observe S', R
        next_state, reward, done, _, _ = env.step(action)
        next_state = get_s(next_state)

        # if S' is terminal then v S' = 0
        target = reward + g * V[next_state] if not done else reward
        TD_error = target - V[state]
        V[state] += alpha_critic * TD_error

        # Update the eligibility traces for the critic and actor
        z_critic[state] += 1
        z_actor[state, action] += 1

        # Update
        grad_log_policy = z_actor / (np.sum(z_actor, axis=1, keepdims=True) + 1e-8)
        policy[state] += alpha_actor * TD_error * grad_log_policy[state]

        # over-time decay of the traces
        z_critic *= g
        z_actor *= g

        # S <- S'
        state = next_state

        history.append(reward)


    episode_reward = sum(history)
    rewards_history.append(episode_reward)
    print(
        f"Episode {episode}: reward={episode_reward} | rolling avg of last 10 epsodes: {np.mean(rewards_history[-10:])}")
