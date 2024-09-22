import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = gym.make("CartPole-v1")

# Setting up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### ~~~~~~ REPLAY MEMORY ~~~~~~ ###
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

### ~~~~~~ C51 DQN ALGORITHM ~~~~~~ ###
class C51DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(C51DQN, self).__init__()

        self.N = N
        self.Vmin = -10
        self.Vmax = 10
        self.delta_z = (self.Vmax - self.Vmin) / (self.N - 1)
        self.z = torch.linspace(self.Vmin, self.Vmax, self.N)

        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions * N)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x).view(-1, n_actions, self.N)

    def get_q_values(self, state):
        # Get the expected Q values by summing over the distributions
        return torch.sum(F.softmax(self(state), dim=2) * self.z, dim=2)

### ~~~~~~ Hyperparameters and utilities Training ~~~~~~ ###

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer

N = 51 # Number of atoms
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = C51DQN(n_observations, n_actions).to(device)
target_net = C51DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            q_values = policy_net(state)
            action_distribution = F.softmax(q_values, dim=-1)
            action = torch.multinomial(action_distribution.view(-1), 1).item()
            return torch.tensor(action, device=device, dtype=torch.long).clamp(0, env.action_space.n - 1).view(1, 1)
            # return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

### ~~~~~~ TRAINING LOOP ~~~~~~ ###
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # This converts batch-array of Transitions to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Computing a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a)
    # Getting distributions for actions taken in state_batch

    # Compute the current state's distribution: shape [batch_size, num_actions, N]
    state_action_values = policy_net(state_batch)

    # Select the distributions for the chosen actions: shape [batch_size, N]
    action_indices = action_batch.unsqueeze(-1).expand(-1, -1, policy_net.N)
    chosen_state_action_values = state_action_values.gather(1, action_indices).squeeze(1)

    # Compute V(s_{t+1}) for all next states using target network
    next_state_values = torch.zeros(BATCH_SIZE, policy_net.N, device=device)

    # Compute a distribution for non-final next states
    if non_final_mask.sum() > 0:
        non_final_next_distributions = F.softmax(target_net(non_final_next_states), dim=2)
        best_next_actions = non_final_next_distributions * policy_net.z
        best_next_actions = best_next_actions.sum(dim=2).max(1)[1].unsqueeze(-1).unsqueeze(-1).expand(non_final_mask.sum(), 1, policy_net.N)
        best_next_distributions = non_final_next_distributions.gather(1, best_next_actions).squeeze(1)

        updated_next_state_values = torch.zeros_like(next_state_values)
        non_final_indices = non_final_mask.nonzero(as_tuple=True)[0]

        # Bellman update: project the future reward distribution
        for idx, non_final_next_state in enumerate(non_final_next_states):
            tz_j = reward_batch[idx] + GAMMA * policy_net.z
            tz_j = tz_j.clamp(min=policy_net.Vmin, max=policy_net.Vmax)
            b_j = (tz_j - policy_net.Vmin) / policy_net.delta_z
            l = b_j.floor().long()
            u = b_j.ceil().long()

            l = l.clamp(min=0, max=policy_net.N - 1)
            u = u.clamp(min=0, max=policy_net.N - 1)
            
            # print("Shape of next_state_values:", next_state_values.shape)
            # print("Shape of l:", l.shape)
            # print("Shape of u:", u.shape)
            # print("Shape of best_next_distributions:", best_next_distributions.shape)
            # print("Shape of (u.float() - b_j):", (u.float() - b_j).shape)
            # print("Shape of best_next_distributions * (u.float() - b_j):", 
            #     (best_next_distributions * (u.float() - b_j)).shape)

            for atom_index in range(policy_net.N):
                l_index = l[atom_index]
                u_index = u[atom_index]
                next_state_values[idx, l_index] += best_next_distributions[idx, atom_index] * (u[atom_index].float() - b_j[atom_index])
                next_state_values[idx, u_index] += best_next_distributions[idx, atom_index] * (b_j[atom_index] - l[atom_index].float())

        #next_state_values = updated_next_state_values

    # Computing the cross entropy loss between projected distributions and model distributions
    log_p = torch.log(chosen_state_action_values + 1e-8)
    loss = -torch.sum(next_state_values * log_p, dim=1).mean()

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    # Update the target network
    # Clipping gradients is common for stability
    with torch.no_grad():
        for target_param, local_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(TAU*local_param.data + (1.0-TAU)*target_param.data)

  ### ~~~~~~ RUNNING IT ~~~~~~ ###
if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 300

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()