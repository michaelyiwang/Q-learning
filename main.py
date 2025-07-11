import random

import gymnasium as gym
import numpy as np

env = gym.make("Taxi-v3")

alpha = 0.9
gamma = 0.95
epsilon = 1.0  # al random
epsilon_decay = 0.9995
min_epsilon = 0.01
num_episodes = 10000
max_steps = 100

q_table = np.zeros((env.observation_space.n, env.action_space.n))

def choose_action(state):
    # using the epsilon-greedy strategy
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state, :])


# --------------------- Training -----------------------
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False

    for step in range(max_steps):
        # select action
        action = choose_action(state)
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
        # end the episode when the episode is finished
        if done or truncated:
            break

    # update exploration rate in the end of each episode
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

# --------------------- Testing -----------------------
env = gym.make(id="Taxi-v3", render_mode="human")
for episode in range(5):
    # reset the environment for a new episode
    state, _ = env.reset()
    done = False

    print(f"Current episode: {episode}")

    for step in range(max_steps):
        # Display the Taxi world
        env.render()
        # select the best action (according to the learned q-table)
        action = np.argmax(q_table[state, :])
        # apply action
        next_state, reward, done, truncated, info = env.step(action)
        # update state after action is taken
        state = next_state
        # end the episode when the episode is finished
        if done or truncated:
            print(f"FINISHED! Episode: {episode}, Reward: {reward}")
            break

# close the window in the end
env.close()
