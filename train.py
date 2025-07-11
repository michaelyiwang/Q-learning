import pickle
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("Taxi-v3")

# define the hyperparameters for the Q-learning algorithm
alpha = 0.9
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.9995
min_epsilon = 0.01
num_episodes = 10000
max_steps = 100

# initialize the Q-table with zeros
q_table = np.zeros((env.observation_space.n, env.action_space.n))


def choose_action(state: int, epsilon: float) -> int:
    """
    Choose an action based on the epsilon-greedy strategy.
    With probability epsilon, choose a random action; otherwise,
    choose the action with the highest Q-value in the current q-table.
    """
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state, :])


# initialize total rewards
total_episodes_rewards = []
# start the training process
for episode in range(num_episodes):
    # reset the environment for a new episode
    state, _ = env.reset()
    done = False
    current_episode_rewards = 0

    for step in range(max_steps):
        # select action
        action = choose_action(state, epsilon)
        # apply action
        next_state, reward, done, truncated, _ = env.step(action)
        # update q table
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state, :])
        q_table[state, action] = (1 - alpha) * old_value + alpha * (
            reward + gamma * next_max
        )
        # update state after action is taken
        state = next_state
        # update rewards
        current_episode_rewards += reward
        # end the episode when the episode is finished
        if done or truncated:
            break

    # update exploration rate in the end of each episode
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    # update the total rewards
    total_episodes_rewards.append(current_episode_rewards)


# store the updated q-table
with open("q_table.pkl", "wb") as f:
    pickle.dump(q_table, f)

# visualize rewards over episodes
plt.plot(total_episodes_rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Q-learning Taxi-v3: Reward per episode")
plt.grid(True)
plt.show()
