from ray import tune
from gym import spaces
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from for_sale_env_r1.fs_first_round import env as fs_env
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG as ppo_config
import copy
import supersuit as ss
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray import shutdown
from ray.rllib.utils.torch_utils import FLOAT_MIN, FLOAT_MAX
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

class DQN(TorchModelV2, nn.Module):
    def __init__(self, observational_space, action_spaces, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, observational_space, action_spaces, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        self.model = nn.Sequential(
            nn.Linear(42, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU())
        self.policy_fn = nn.Linear(512, num_outputs)
        self.value_fn = nn.Linear(512, 1)


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



def env_creator(args):
    env = fs_env()
    # env = ss.color_reduction_v0(env, mode='B')
    # env = ss.dtype_v0(env, 'float32')
    # env = ss.resize_v0(env, x_size=84, y_size=84)
    # env = ss.frame_stack_v1(env, 3)
    # env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    return env


if __name__ == "__main__":

    
    shutdown()

    env_name = "fs_1r_3_rounds_game_ICM_v0"

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
    
        # Trying to use the ICM model
    config = ppo_config.copy()
    config["env"] = env_name
    config["framework"] = "torch"
    config["num_workers"] = 0
    config["multiagent"] = {
                "policies": policies,
                "policy_mapping_fn": (
                    lambda agent_id: policy_ids[0]),
            }
    config["exploration_config"] = {
    "type": "Curiosity",  # <- Use the Curiosity module for exploring.
    "eta": 1.0,  # Weight for intrinsic rewards before being added to extrinsic ones.
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
    "beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
    # Specify, which exploration sub-type to use (usually, the algo's "default"
    # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
    "sub_exploration": {
        "type": "StochasticSampling",
    }
}

    print("pre tune.run")
    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 5000000},
        checkpoint_freq=10,
        local_dir="~/ray_results/"+env_name,
        config=config,
    )
    print("post tune.run")