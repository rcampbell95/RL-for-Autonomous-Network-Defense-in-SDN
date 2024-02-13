from rl_autonomous_defence.callbacks import SelfPlayCallback
from ray.rllib.policy.policy import PolicySpec
import os
from random_policy import RandomPolicy

from rl_autonomous_defence.utils import select_policy, string_to_bool
from rl_autonomous_defence.agent_config import ATTACKER_CONFIG, DEFENDER_CONFIG
from rl_autonomous_defence.train_config import train_config

config = {
    "env": "AutonomousDefenceEnv",
    "callbacks": SelfPlayCallback,
    "log_level": "DEBUG",
    "num_gpus": train_config["train"]["num_gpus"],
    "rollout_fragment_length": 200,
    "train_batch_size": train_config["train"]["train_batch_size"],#os.getenv("RL_SDN_BATCHSIZE", 1000),
    "timesteps_per_iteration": 10000,
    "sgd_minibatch_size": 512,
    "clip_rewards": 200,
    "num_workers": 10,
    "num_envs_per_worker": 1,
    "horizon": train_config["train"]["horizon"],
    "remote_worker_envs": False,
    "num_cpus_for_driver": 1,
    "num_gpus_per_worker": 0,
    "lr": train_config["train"]["lr"],
    "lr_schedule": [
        [0, train_config["train"]["lr"]],
        [train_config["environment"]["timesteps"], 1e-5]
    ],
    "framework": "tf2",
    "tf_session_args": {
        "device_count": {
            "CPU": 1,
            "GPU": 1
        }
    },
    #"evaluation_config": {
    #    "multiagent": {
    #        "policy_mapping_fn": lambda x: x
    #    }
        # Store videos in this relative directory here inside
        # the default output dir (~/ray_results/...).
        # Alternatively, you can specify an absolute path.
        # Set to True for using the default output dir (~/ray_results/...).
        # Set to False for not recording anything.
    #    "record_env": "videos",
        # "record_env": "/Users/xyz/my_videos/",

        # Render the env while evaluating.
        # Note that this will always only render the 1st RolloutWorker's
        # env and only the 1st sub-env in a vectorized env.
    #    "render_env": False
    #},
    #"evaluation_duration_unit": "timesteps",
    "multiagent": {
            "policies_to_train": ["attacker", "defender"],
            "policies": {
                "attacker": PolicySpec(config=ATTACKER_CONFIG),
                "defender_v0": PolicySpec(policy_class=DEFENDER_CONFIG),
                "attacker_v0": PolicySpec(policy_class=ATTACKER_CONFIG),
                "defender": PolicySpec(config=DEFENDER_CONFIG),
            },
            "policy_mapping_fn": select_policy,
        },
    }

if string_to_bool(os.getenv("RL_SDN_EVALUATE", False)):
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id == "attacker":
            return "attacker"
        elif agent_id == "defender":
            return "defender"

    config["evaluation_interval"] = 1
    config["evaluation_duration"] = 10
    config["evaluation_duration_unit"] = "episodes"
    config["evaluation_config"] = {
        "multiagent": {
            "policy_mapping_fn": policy_mapping_fn
        }
    }
    config["always_attach_evaluation_results"] = True,
