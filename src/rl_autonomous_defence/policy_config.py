from rl_autonomous_defence.callbacks import SelfPlayCallback
from ray.rllib.policy.policy import PolicySpec
import os
from random_policy import RandomPolicy

from rl_autonomous_defence.utils import select_policy, string_to_bool
from rl_autonomous_defence.agent_config import ATTACKER_CONFIG, DEFENDER_CONFIG


config = {
    "env": "AutonomousDefenceEnv",
    "callbacks": SelfPlayCallback,
    "log_level": "DEBUG",
    "num_gpus": float(os.environ.get("RLLIB_NUM_GPUS", "1")),
    "rollout_fragment_length": 200,
    "train_batch_size": os.getenv("RL_SDN_BATCHSIZE", 5000),
    "timesteps_per_iteration": 10000,
    "sgd_minibatch_size": 50,
    "clip_rewards": 200,
    "num_workers": 16,
    "num_envs_per_worker": 50,
    "horizon": int(os.environ.get("RL_SDN_HORIZON", 200)),
    "remote_worker_envs": False,
    "num_cpus_for_driver": 4,
    "num_gpus_per_worker": 0,
    "lr": float(os.getenv("RL_SDN_LR", 3e-4)),
    "lr_schedule": [
        [0, float(os.getenv("RL_SDN_LR", 3e-4))],
        [int(os.getenv("RL_SDN_TIMESTEPS", "45000").strip()), 1e-5]
    ],
    #"min_train_timesteps_per_reporting": 5000,
    #"batch_mode": "complete_episodes",
    "framework": "tfe",
    # Evaluate once per training iteration.
    #"evaluation_interval": 0,
    # Run evaluation on (at least) two episodes
    #"evaluation_duration": 100,
    #"evaluation_duration_unit": "episodes",
    #"evaluation_parallel_to_training": True,
    # ... using one evaluation worker (setting this to 0 will cause
    # evaluation to run on the local evaluation worker, blocking
    # training until evaluation is done).
    #"always_attach_evaluation_results": True,
    #"evaluation_num_workers": 0,
    "tf_session_args": {
        "device_count": {
            "CPU": 2,
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
                "defender_v0": PolicySpec(policy_class=RandomPolicy),
                "attacker_v0": PolicySpec(policy_class=RandomPolicy),
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
