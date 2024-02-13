import os
import json
import copy

import scipy.stats as sp

from ray.tune.registry import register_env

# import the pettingzoo environment
# import rllib pettingzoo interface
from ray.rllib.env import PettingZooEnv
import ray
from ray import tune

from ray.tune.schedulers import ASHAScheduler

from ray.rllib.policy.policy import PolicySpec

from rl_autonomous_defence.utils import select_policy

import tensorflow as tf

from rl_autonomous_defence import ade
from rl_autonomous_defence.random_policy import RandomPolicy
from rl_autonomous_defence import policy_config
from rl_autonomous_defence.CheckpointWrapper import CheckpointWrapperPPO
from rl_autonomous_defence.train_config import train_config
from rl_autonomous_defence.agent_config import ATTACKER_CONFIG, DEFENDER_CONFIG



def gen_gamma():
    gen = sp.skewnorm(.01, loc=.95, scale=.05)
    return min(max(gen.rvs(1).item(), 0), .999)



if __name__ == "__main__":
    ray.init(log_to_driver=False, num_cpus=16, num_gpus=1)

    register_env('AutonomousDefenceEnv', lambda x: PettingZooEnv(ade.env()))
    hyperparam_config = {
        "env": "AutonomousDefenceEnv",
        "log_level": "DEBUG",
        "num_workers": 2,
        "num_gpus": 0,
        "num_envs_per_worker": 5,
        "horizon": 200,
        "lr": 0.001,
        "sgd_minibatch_size": 2000,
        "train_batch_size": 10000,
    }

    ATTACKER_CONFIG.update(
        {
            "lambda": 0.95,
            "clip_param": 0.1,
            "gamma": 0.99,
            "vf_loss_coeff": 0.3, 
            "entropy_coeff": 0.0,
            "kl_coeff": 0.8,
            "num_iter_sgd": 10,
            "model": {
                "custom_model": "gcn_attacker_action_mask_model",
                "custom_model_config": {
                    "network_size": train_config["environment"]["network_size"],
                    "masked_actions": True,
                    "gcn_hiddens": [32],
                    "dense_hiddens": 64,
                    "num_frames": 2,
                    "reward": train_config["environment"]["reward"]
                }
            }
        }
    )

    DEFENDER_CONFIG.update(
        {
            "lambda": 0.95,
            "clip_param": 0.1,
            "gamma": 0.99,
            "vf_loss_coeff": 0.7, #tune.choice([(1e-1 * i) for i in range(10)]), 
            "entropy_coeff": 0.0, #tune.choice([(1e-1 * i) for i in range(10)]),
            "kl_coeff": 0.3, #tune.choice([(1e-1 * i) for i in range(10)]),
            "num_iter_sgd": 10,
            "model": {
                "custom_model": "gcn_defender_action_mask_model",
                "custom_model_config": {
                    "network_size": train_config["environment"]["network_size"],
                    "mis": train_config["environment"]["mean_impact_score"],
                    "mes": train_config["environment"]["mean_exploitability_score"],
                    "masked_actions": True,
                    "gcn_hiddens": [32],
                    "dense_hiddens": 64,
                    "num_frames": 2,
                    "reward": train_config["environment"]["reward"]
                }
            }
        }
    )

    ATTACKER_V0_CONFIG = copy.deepcopy(ATTACKER_CONFIG)
    ATTACKER_V0_CONFIG["model"]["custom_model_config"]["masked_actions"] = False

    DEFENDER_V0_CONFIG = copy.deepcopy(DEFENDER_CONFIG)
    DEFENDER_V0_CONFIG["model"]["custom_model_config"]["masked_actions"] = False

    policy_config.config.update(hyperparam_config)

    assert ATTACKER_CONFIG["model"]["custom_model_config"]["masked_actions"] == True

    policy_config.config["multiagent"] = {
        "policies_to_train": ["attacker", "defender"],
        "policies": {
            "attacker": PolicySpec(config=ATTACKER_CONFIG),
            "defender_v0": PolicySpec(config=DEFENDER_V0_CONFIG),
            "attacker_v0": PolicySpec(config=ATTACKER_V0_CONFIG),
            "defender": PolicySpec(config=DEFENDER_CONFIG),
        },
        "policy_mapping_fn": select_policy,

    }

    stop = {
        "timesteps_total": 5e6
    }

    try:
        results = tune.run(CheckpointWrapperPPO,
                        config=policy_config.config, 
                        name="reward",
                        verbose=1,
                        metric="episode_len_mean",
                        mode="min",
                        num_samples=5,
                        checkpoint_freq=1,
                        checkpoint_at_end=True,
                        local_dir=train_config["experiment_dir"],
                        stop=stop)
                        #num_samples=10)
    except Exception as e:
        raise(e)
    finally:
        ray.shutdown()
