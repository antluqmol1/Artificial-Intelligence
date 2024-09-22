#pip install gym        # The gym module is a part of OpenAI Gym, 
                        # a toolkit for developing and comparing reinforcement learning algorithms.

import gym
env = gym.make("CartPole-v1", render_mode="human")

env.observation_space.sample()
nb_actions = env.action_space.n

episodes = 20
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    
    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, truncated, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))