import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from for_sale_env_r2.fs_second_round import env as fs_env

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
    
plt.ion()
    
FLOAT_MIN = -3.4e38
FLOAT_MAX = 3.4e38
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, observational_space, num_outputs):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(474, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU())
        self.policy_fn = nn.Linear(512, num_outputs)
        self.value_fn = nn.Linear(512, 1)


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, input_dict):
        action_mask = torch.from_numpy(input_dict["action_mask"]).to(device)
        x = torch.from_numpy(input_dict["observation"]).to(device)
        model_out = self.model(x)
        self._value_out = self.value_fn(model_out)
        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX).to(device)
        # print(self.policy_fn(model_out) + inf_mask)
        return self.policy_fn(model_out) + inf_mask

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()

env = fs_env(rounds=3, static_hands=False, static_draw=True)
env.reset()
observation, reward, done, info = env.last()

# Get number of actions from gym action space
n_actions = env.action_space("player_0").n
observation_space = env.observation_space("player_0")


policy_net = DQN(observation_space, n_actions).to(device)
target_net = DQN(observation_space, n_actions).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = {agent: ReplayMemory(10000) for agent in env.agents}


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            
            # print("policy_net: ",policy_net)
            # print("state: ",state)
            
            return policy_net(state).argmax().view(1, 1)
    else:
        # print("sample less than eps_treshold")
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


wins = {"player_0": [], 
        "player_1": [],
        "player_2": []}
draws_wins = []
    
def plot_wins(episode):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('wins')
    # if episode < 40000:
    if False:
        x = np.arange(0, episode + 1)
        # plt.plot(x, pd.Dataframe({"player_0": np.cumsum(wins["player_0"])}).ewm(span=1000, adjust=False).mean(), color='b', label='player_0')
        # plt.plot(x, pd.Dataframe({"player_1": np.cumsum(wins["player_1"])}).ewm(span=1000, adjust=False).mean(), color='g', label='player_1')
        # plt.plot(x, pd.Dataframe({"player_2": np.cumsum(wins["player_2"])}).ewm(span=1000, adjust=False).mean(), color='r', label='player_2')
        plt.plot(x, np.cumsum(wins["player_0"]), color='b', label='player_0')
        plt.plot(x, np.cumsum(wins["player_1"]), color='g', label='player_1')
        plt.plot(x, np.cumsum(wins["player_2"]), color='r', label='player_2')
        plt.plot(x, np.cumsum(draws_wins), color='y', label='draws')

    else:
        # x = np.arange(episode - 10000, episode+1)
        x = np.arange(0, episode+1)
        
        plt.plot(x, pd.DataFrame({"player_0": np.cumsum(wins["player_0"])}).ewm(alpha=0.8).mean(), color='b', label='player_0')
        plt.plot(x, pd.DataFrame({"player_1": np.cumsum(wins["player_1"])}).ewm(alpha=0.8).mean(), color='g', label='player_1')
        plt.plot(x, pd.DataFrame({"player_2": np.cumsum(wins["player_2"])}).ewm(alpha=0.8).mean(), color='r', label='player_2')

        # plt.plot(x, np.cumsum(wins["player_0"])[-10000:], color='b', label='player_0')
        # plt.plot(x, np.cumsum(wins["player_1"])[-10000:], color='g', label='player_1')
        # plt.plot(x, np.cumsum(wins["player_2"])[-10000:], color='r', label='player_2')
        # plt.plot(x, np.cumsum(draws_wins)[-10000:], color='y', label='draws')

    

    # plt.pause(1)  # pause a bit so that plots are updated
    if is_ipython:
        # pass
        # display.clear_output(wait=True)
        # display.display(plt.gcf())
        plt.show()
    else:
        plt.savefig("/home/jaoi/forsale_data/dqn/four_layer"+ str(episode))

def optimize_model(agent):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory[agent].sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

num_episodes = 1000001
env = fs_env(rounds=3, static_hands=False, static_draw=True)
# print("=*40 first init ", env._has_updated)
env.reset()
# print("=*40 first reset ", env._has_updated)

observation, reward, done, info = env.last()
# print("=*40 first last", env._has_updated)
# print(observation)



for i_episode in tqdm.tqdm(range(num_episodes)):
    # Initialize the environment and state
    prev_observation = {"player_0": False, 
                        "player_1": False, 
                        "player_2": False}
    prev_action = {"player_0": False, 
                   "player_1": False,
                   "player_2": False}
    # print("=*40 pre second reset", env._has_updated)
    
    env.reset()
    # print("=*40 first init", env._has_updated)
    # env._has_updated = True
    added = False
    for t, agent in enumerate(env.agent_iter()):
        # print(agent)
        
        observation, reward, done, info = env.last()
        
        reward = torch.tensor([reward], device=device)

        # Store the transition in memory
        if prev_observation[agent]:
            memory[agent].push(prev_observation, prev_action[agent], observation, reward)
            
        # Select and perform an action
        if not done:
            action = select_action(observation)
        else:
            action = None
            
        # if there is a winner, append to list
        if reward.item() == 1:
            for a,l in wins.items():
                if a == agent:
                    l.append(1)
                else:
                    l.append(0)
            draws_wins.append(0)
        elif done and reward.item() == 0 and not added:
            draws_wins.append(1)
            for a,l in wins.items():
                l.append(0)
            added = True
        
        # Perform one step of the optimization (on the policy network)
        optimize_model(agent)
        

            
        # Move to the next state
        prev_observation[agent] = observation
        if torch.is_tensor(action):
            prev_action[agent] = action.item()
            # print(action)
            # print(action.item())
            # print(observation["action_mask"])
            env.step(action.item())
        else:
            # print("*"*50)
            # print("action is: ", action)
            # print("agent is: ", agent)
            # print(env.render())
            env.step(action)
            
    if i_episode != 0 and i_episode % 50000 == 0:
        plot_wins(i_episode)
        torch.save(target_net.state_dict(), "/home/jaoi/forsale_data/dqn/four_layer" + str(i_episode))
    
        
        
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
