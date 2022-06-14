from math import perm
import pickle
from re import A
from tabnanny import check
import os
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG as ppo_config
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from for_sale_env_r1.fs_first_round import env as fs_env
from ray.rllib.utils.torch_utils import FLOAT_MIN, FLOAT_MAX
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import random
import numpy as np
from collections import defaultdict, Counter
import tqdm
from itertools import permutations


torch, nn = try_import_torch()

rounds = 4
stack_number = int((3 * rounds) * 1.25)
observation_shape = (6 + rounds) * 3 + 4 + stack_number

class DQN(TorchModelV2, nn.Module):
    def __init__(self, observational_space, action_spaces, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, observational_space, action_spaces, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        self.model = nn.Sequential(
            nn.Linear(observation_shape, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU())
        self.policy_fn = nn.Linear(256, num_outputs)
        self.value_fn = nn.Linear(256, 1)


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, input_dict, state, seq_lens):
        action_mask = input_dict["obs"]["action_mask"]
        x = input_dict["obs"]["observation"]
        model_out = self.model(x)
        self._value_out = self.value_fn(model_out)
        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)
        # print(self.policy_fn(model_out) + inf_mask)
        return self.policy_fn(model_out) + inf_mask, state
    def value_function(self):
        return self._value_out.flatten()

# class DQN(TorchModelV2, nn.Module):
#     def __init__(self, observational_space, action_spaces, num_outputs, *args, **kwargs):
#         TorchModelV2.__init__(self, observational_space, action_spaces, num_outputs, *args, **kwargs)
#         nn.Module.__init__(self)
#         self.model = nn.Sequential(
#             nn.Linear(42, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU())
#         self.policy_fn = nn.Linear(256, num_outputs)
#         self.value_fn = nn.Linear(256, 1)


#     # Called with either one element to determine next action, or a batch
#     # during optimization. Returns tensor([[left0exp,right0exp]...]).
#     def forward(self, input_dict, state, seq_lens):
#         action_mask = input_dict["obs"]["action_mask"]
#         x = input_dict["obs"]["observation"]
#         model_out = self.model(x)
#         self._value_out = self.value_fn(model_out)
#         inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)
#         # print(self.policy_fn(model_out) + inf_mask)
#         return self.policy_fn(model_out) + inf_mask, state

#     def value_function(self):
#         return self._value_out.flatten()


def env_creator(args):
    env = fs_env(rounds=4)
    # env = ss.color_reduction_v0(env, mode='B')
    # env = ss.dtype_v0(env, 'float32')
    # env = ss.resize_v0(env, x_size=84, y_size=84)
    # env = ss.frame_stack_v1(env, 3)
    # env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    return env


env_name = "fs_1r_4_rounds_game_PPO_v0"

register_env(env_name, lambda config: PettingZooEnv(env_creator(config)))

test_env = PettingZooEnv(env_creator({}))
obs_space = test_env.observation_space
act_space = test_env.action_space


ModelCatalog.register_custom_model("DQN", DQN)


def gen_policy(i):
    config = {
        "model": {
            "custom_model": "DQN",
        },
        "gamma": 0.99,
    }
    return (None, obs_space, act_space, config)


policies = {"policy_0": gen_policy(0)}

policy_ids = list(policies.keys())

config = ppo_config.copy()
config["env"] = env_name
config["framework"] = "torch"
config["num_workers"] = 0
config["multiagent"] = {
    "policies": policies,
    "policy_mapping_fn": (
        lambda agent_id: policy_ids[0]),
}

config_icm = ppo_config.copy()
config_icm["env"] = env_name
config_icm["framework"] = "torch"
config_icm["num_workers"] = 0
config_icm["multiagent"] = {
    "policies": policies,
    "policy_mapping_fn": (
        lambda agent_id: policy_ids[0]),
}
config_icm["exploration_config"] = {
    "type": "Curiosity",  # <- Use the Curiosity module for exploring.
    # Weight for intrinsic rewards before being added to extrinsic ones.
    "eta": 1.0,
    "lr": 0.001,  # Learning rate of the curiosity (ICM) module.
    "feature_dim": 288,  # Dimensionality of the generated feature vectors.
    # Setup of the feature net (used to encode observations into feature (latent) vectors).
    "feature_net_config": {
        "fcnet_hiddens": [],
        "fcnet_activation": "relu",
    },
    "inverse_net_hiddens": [256],  # Hidden layers of the "inverse" model.
    "inverse_net_activation": "relu",  # Activation of the "inverse" model.
    "forward_net_hiddens": [256],  # Hidden layers of the "forward" model.
    "forward_net_activation": "relu",  # Activation of the "forward" model.
    # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
    "beta": 0.2,
    # Specify, which exploration sub-type to use (usually, the algo's "default"
    # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
    "sub_exploration": {
        "type": "EpsilonGreedy",
    }
}


# checkpoint_file = "/home/jaoi/ray_results/fs_1r_3_rounds_game_ICM_v1/PPO/PPO_fs_1r_3_rounds_game_ICM_v1_4fe38_00000_0_2022-04-22_13-14-01/checkpoint_012090/checkpoint-12090"
# checkpoint_file = "/home/jaoi/imported_models/PPO3r_v0/checkpoint_025000/checkpoint-25000"
checkpoint_file = "/home/jaoi/imported_models/PPO4r_v0/checkpoint_025000/checkpoint-25000"

checkpoint_file_1 = "/home/jaoi/imported_models/PPO4r_v1/checkpoint_025000/checkpoint-25000"
checkpoint_file_2 = "/home/jaoi/imported_models/ICM4r_v0/checkpoint_025000/checkpoint-25000"

if os.path.isfile(checkpoint_file):
    print("its here")
else:
    print("its not here")

# with open(checkpoint_file, "rb") as fp:
#     d = pickle.load(fp)
#     print(type(d))
#     print(list(d.keys()))
#     print(list(d["train_exec_impl"]["info"]
#           ["learner"]["policy_0"]["model"].keys()))
#     print(type(d["train_exec_impl"]["info"]["learner"]["policy_0"]["model"]))
def rand_policy(observation, agent):
    action = random.choice(np.flatnonzero(observation['action_mask']))
    return action

# player_models = {}
# player_models["ICMv0"] = PPOTrainer(config=config_icm)
# player_models["ICMv0"].restore(checkpoint_file_2)
# player_models["PPOv0"] = PPOTrainer(config=config)
# player_models["PPOv0"].restore(checkpoint_file)
# player_models["PPOv1"] = PPOTrainer(config=config)
# player_models["PPOv1"].restore(checkpoint_file_1)


# model_names = ["ICMv0", "PPOv0", "PPOv1"]
# players_perms = permutations(model_names)

def compare_agents(model_names, player_models, env, games):
    model_perms = permutations(model_names)
    env.reset()
    agent_list = list(env.agents)
    for perm in model_perms:
        agent_to_model_dict = {agent_list[i]: perm[i] for i in range(len(agent_list))}
        games_to_save = []
        c = Counter()
        for game in tqdm.tqdm(range(games)):
            g = defaultdict(list)
            env.reset()

            winning_models = []
            for agent in env.agent_iter():
                observation, reward, done, info = env.last()
                if done:
                    action = None
                elif agent_to_model_dict[agent] == "ICMv0":
                    # action = rand_policy(observation, agent)
                    action, _, _ = player_models["PPOv0"].get_policy("policy_0").compute_single_action(observation)
                else:
                    action, _, _ = player_models[agent_to_model_dict[agent]].get_policy("policy_0").compute_single_action(observation)

                if action != None:
                    g[agent_to_model_dict[agent]].append((observation["observation"], action))
                

                env.step(action)
                if reward == 1:
                    winning_models.append(agent_to_model_dict[agent])
                    c[agent_to_model_dict[agent]] += 1
            
            modelstat_to_save = random.choice(winning_models)
            games_to_save.extend(g[modelstat_to_save])
        print(agent_to_model_dict)
        print(c)

    

if __name__ == "__main__":
    
    # config = ppo_config.copy()
    # config["env"] = env_name
    # config["framework"] = "torch"
    PPOagent = PPOTrainer(config=config_icm)
    PPOagent.restore(checkpoint_file_2)
    env = fs_env(rounds=4)
    smart_agent = "player"

    env.reset()
    
    games = 100000
    # compare_agents(model_names, player_models, env, games)
    games_to_save = []
    c = Counter()
    for game in tqdm.tqdm(range(games)):
        g = defaultdict(list)
        env.reset()

        winning_agents = []
        for agent in env.agent_iter():
            observation, reward, done, info = env.last()
            if done:
                action = None
                # env.render()
            # elif agent != smart_agent:
                # action = rand_policy(observation, agent)
                # action, _, _ = PPOagent.get_policy("policy_0").compute_single_action(observation)
                # print(observation["observation"])
                # env.render()
                
                # action = int(input())
                # while action not in observation["action_mask"]:
                #     print("you can't do {}".format(action))
                #     action = int(input())

            else:
                # action, _, _ = player_models["ICMv0"].get_policy("policy_0").compute_single_action(observation)
                action, _, _ = PPOagent.get_policy("policy_0").compute_single_action(observation)


            if action != None:
                g[agent].append((observation["observation"], action))
            

            env.step(action)
            if reward == 1:
                winning_agents.append(agent)
                c[agent] += 1
        
        agent_to_save = random.choice(winning_agents)
        games_to_save.extend(g[agent_to_save])
    print(c)

    with open("/home/jaoi/master22/pet_for_sale/winning_games_db/4ICM_{}_games.pkl".format(games), "wb") as fp:
        pickle.dump(np.array(games_to_save, dtype=object), fp)
    

        # print(agent, action)
    # print(agent.get_default_policy_class().get_weights())
    # agent.compute_action(test_env.reset())
    # print(agent.compute_single_action(test_env.reset()["player_0"]))
    
