from ray import tune
from gym import spaces
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from for_sale_env_r2.fs_second_round import env as fs_env
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
            nn.Linear(474, 256),
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
    env = fs_env(static_hands=False, static_draw=False, rounds=8)
    # env = ss.color_reduction_v0(env, mode='B')
    # env = ss.dtype_v0(env, 'float32')
    # env = ss.resize_v0(env, x_size=84, y_size=84)
    # env = ss.frame_stack_v1(env, 3)
    # env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    return env


if __name__ == "__main__":
    shutdown()

    env_name = "fs_2r_full_game"

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
    print("pre tune.run")
    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 5000000 * 25},
        checkpoint_freq=10,
        local_dir="~/ray_results/"+env_name,
        config={
            # Environment specific
            "env": env_name,
            # General
            "log_level": "ERROR",
            "framework": "torch",
            "num_gpus": 1,
            "num_workers": 4,
            "num_envs_per_worker": 1,
            "compress_observations": False,
            "batch_mode": 'truncate_episodes',

            # 'use_critic': True,
            'use_gae': True,
            "lambda": 0.9,

            "gamma": .99,

            # "kl_coeff": 0.001,
            # "kl_target": 1000.,
            "clip_param": 0.4,
            'grad_clip': None,
            "entropy_coeff": 0.1,
            'vf_loss_coeff': 0.25,

            "sgd_minibatch_size": 64,
            "num_sgd_iter": 10, # epoc
            'rollout_fragment_length': 512,
            "train_batch_size": 512,
            'lr': 2e-05,
            "clip_actions": True,

            # Method specific
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": (
                    lambda agent_id: policy_ids[0]),
            },
        },
    )
    print("post tune.run")