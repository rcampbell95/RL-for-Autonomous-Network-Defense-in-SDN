import ray.rllib.agents.pg as PG
#import ray.rllib.agents. as PG

from ray.tune.registry import register_env
# import the pettingzoo environment
# import rllib pettingzoo interface
from ray.rllib.env import PettingZooEnv
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.agents.pg import PGTrainer
import os
from custom_metrics import CustomMetricsCallback

import logging

from ray import tune
import ade

def select_policy(agent_id):
    return agent_id

env_creator = lambda config: ade.env()
register_env('AutonomousDefenceEnv', lambda config: PettingZooEnv(env_creator(config)))


stop = {
    "training_iteration": 150,
    "timesteps_total": 200000,
    "episode_reward_mean": 1000,
}

config = {
    "env": "AutonomousDefenceEnv",
    "callbacks": CustomMetricsCallback,
    "log_level": "DEBUG",
    "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
    "num_workers": 0,
    "num_envs_per_worker": 1,
    "multiagent": {
            "policies_to_train": ["attcker", "defender"],
            "policies": {
                "attacker": PolicySpec(config={
                    "agent_id": 0,
                    "model": {
                        "use_lstm": False
                    },
                    "framework": "tf",
                }),
                 "defender": PolicySpec(config={
                    "agent_id": 1,
                    "model": {
                        "use_lstm": False
                    },
                    "framework": "tf",
                }),
            },
            "policy_mapping_fn": select_policy,
        },
        "framework": "tf"
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

algorithms = ["A2C", "PPO", "PG"]

for algo in algorithms:
    results = tune.run(algo, config=config, stop=stop, verbose=1)
#PGTrainer(config=config).train()
