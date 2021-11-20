import ray.rllib.agents.pg as PG

from ray.tune.registry import register_env
# import the pettingzoo environment
# import rllib pettingzoo interface
from ray.rllib.env import PettingZooEnv
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.agents.pg import PGTrainer
import os

import logging

from ray import tune
import ade

def select_policy(agent_id):
    return agent_id

env_creator = lambda config: ade.env()
register_env('AutonomousDefenceEnv', lambda config: PettingZooEnv(env_creator(config)))


stop = {
    "training_iteration": 150,
    "timesteps_total": 100000,
    "episode_reward_mean": 1000,
}

config = {
    "env": "AutonomousDefenceEnv", 
    "log_level": "DEBUG",
    "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
    "num_workers": 0,
    "num_envs_per_worker": 4,
    "multiagent": {
            "policies_to_train": ["attcker", "defender"],
            "policies": {
                "attacker": PolicySpec(config={
                    "model": {
                        "use_lstm": False
                    },
                    "framework": "tf",
                }),
                 "defender": PolicySpec(config={
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


results = tune.run("PG", config=config, stop=stop, verbose=1)
#PGTrainer(config=config)


