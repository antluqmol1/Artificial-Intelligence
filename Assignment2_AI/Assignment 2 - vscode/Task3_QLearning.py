import gymnasium as gym
import math
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("CartPole-v1")


# Discretize the continuous state space into bins
def discretize_state(state, bins=[18, 14, 18, 14]):
    upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)]
    lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)]
    ratios = [(state[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(state))]
    new_state = [int(round((bins[i] - 1) * ratios[i])) for i in range(len(state))]
    new_state = [min(bins[i] - 1, max(0, new_state[i])) for i in range(len(state))]
    return tuple(new_state)

# Initialize Q-table
q_table = np.zeros([18, 14, 18, 14, env.action_space.n])

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Initial exploration rate
min_epsilon = 0.01  # Minimum exploration rate
epsilon_decay = 0.995  # Exploration decay rate

episode_durations = []  # List to store each episode duration

for i_episode in range(1, 100001):  # Running 1000 episodes for illustration
    initial_state, _ = env.reset()  # Correctly reset the environment to get the initial state
    state = discretize_state(initial_state)  # Discretize the state
    
    epochs, penalties, reward = 0, 0, 0
    done = False
    
    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[state])  # Exploit learned values

        #print(env.step(action))
        next_state, reward, done, extra_boolean, info = env.step(action)
        next_state = discretize_state(next_state)

        old_value = q_table[state + (action,)]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state + (action,)] = new_value

        state = next_state
        epochs += 1
        
        if done and epsilon > min_epsilon:
            epsilon *= epsilon_decay

    episode_durations.append(epochs)  # Append the duration of the episode

    if i_episode % 100 == 0:
        print(f"Episode: {i_episode}")

#print("Training finished.\n")
# Plotting the episode durations
plt.plot(episode_durations)
plt.title('Episode durations over time')
plt.xlabel('Episode')
plt.ylabel('Duration')
plt.show()
