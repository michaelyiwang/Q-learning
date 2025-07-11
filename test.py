import os
import pickle

import gymnasium as gym
import numpy as np

# load up the obtained q-table from training
if os.path.exists("q_table.pkl"):
    with open("q_table.pkl", "rb") as f:
        q_table = pickle.load(f)

# define the maximum number of steps in each episode
max_steps = 100

# create the environment (with visualization)
env = gym.make(id="Taxi-v3", render_mode="human")
for episode in range(5):
    # reset the environment for a new episode
    state, _ = env.reset()
    done = False

    print(f"Current episode: {episode}")

    for step in range(max_steps):
        # Display it
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
