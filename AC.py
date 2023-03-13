import os
import gymnasium as gym
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
rnd = np.random.default_rng(112233)

env = gym.make('CartPole-v1')
env.reset()

# Initialize the policy and value function
num_states = 10 ** len(env.reset()[0])
num_actions = env.action_space.n
policy = np.ones((num_states, num_actions)) / num_actions  # Uniform random policy
V = np.zeros(num_states)

# Set the learning rates and discount factor
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

    # Initialize eligibility traces
    e_critic = np.zeros(num_states)
    e_actor = np.zeros((num_states, num_actions))

    # Initialize the episode history
    history = []

    while not done:
        # Sample an action from the policy
        action_probs = np.exp(policy[state]) / np.sum(np.exp(policy[state]))
        if rnd.random() < e:
            action = rnd.choice(list(range(num_actions)))
        else:
            action = np.random.choice(num_actions, p=action_probs)

        # Take a step in the environment
        next_state, reward, done, _, _ = env.step(action)
        next_state = get_s(next_state)

        # Update the TD error and the value function
        target = reward + g * V[next_state] if not done else reward
        TD_error = target - V[state]
        V[state] += alpha_critic * TD_error

        # Update the eligibility traces for the critic and actor
        e_critic[state] += 1
        e_actor[state, action] += 1

        # Update the policy parameters
        grad_log_policy = e_actor / (
                    np.sum(e_actor, axis=1, keepdims=True) + 1e-8)  # Compute the gradient of the log policy
        policy[state] += alpha_actor * TD_error * grad_log_policy[state]

        # Decay the eligibility traces
        e_critic *= g
        e_actor *= g

        # Update the current state
        state = next_state

        # Save the current state, action, and reward for the episode history
        history.append((state, action, reward))

    # Print the episode reward
    episode_reward = sum([r for (_, _, r) in history])
    rewards_history.append(episode_reward)
    print(
        f"Episode {episode}: reward={episode_reward} | rolling avg of last 10 epsodes: {np.mean(rewards_history[-10:])}")
