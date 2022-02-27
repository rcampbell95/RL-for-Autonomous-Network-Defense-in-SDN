from custom_metrics import CustomMetricsCallback
from ray.rllib.policy.policy import PolicySpec
import os
from ray.rllib.models import ModelCatalog

from utils import select_policy, string_to_bool
from AutoregressiveActionsModel import BinaryAutoregressiveDistribution, AutoregressiveActionModel


#config["partial_obs"] = {"lstm": tune.grid_search([True, False])}

config = {
    "env": "AutonomousDefenceEnv",
    "callbacks": CustomMetricsCallback,
    "log_level": "DEBUG",
    "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
    "rollout_fragment_length": 200,
    "num_workers": 0,
    "num_envs_per_worker": 1,
    "num_cpus_for_driver": 1,
    #"min_train_timesteps_per_reporting": 5000,
    "timesteps_per_iteration": 5000,
    "framework": "tf",
        # Evaluate once per training iteration.
    #"evaluation_interval": None,
    # Run evaluation on (at least) two episodes
    #"evaluation_duration": 2,
    #"evaluation_duration_unit": "episodes",
    #"evaluation_parallel_to_training": True,
    # ... using one evaluation worker (setting this to 0 will cause
    # evaluation to run on the local evaluation worker, blocking
    # training until evaluation is done).
    #"evaluation_num_workers": 1,
    "tf_session_args": {
        "device_count": {
            "CPU": 2
        }
    },
    #"evaluation_config": {
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
                "attacker": PolicySpec(config={
                    "agent_id": 0,
                    "model": {
                        "use_lstm": string_to_bool(os.getenv("RL_SDN_ATTACKERLSTM", False)),
                        "vf_share_layers": False
                    },
                    "framework": "tf",
                }),
                 "defender": PolicySpec(config={
                    "agent_id": 1,
                    "model": {
                        "use_lstm":  string_to_bool(os.getenv("RL_SDN_DEFENDERLSTM", False))
                    },
                    "framework": "tf",
                }),
            },
            "policy_mapping_fn": select_policy,
        },
    }

if string_to_bool(os.getenv("RL_SDN_ICM", False)):
    config["exploration_config"] = {
        "type": "Curiosity",  # <- Use the Curiosity module for exploring.
        "eta": float(os.getenv("RL_SDN_ETA", "0.5").strip()),  # Weight for intrinsic rewards before being added to extrinsic ones.
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
        "beta": float(os.getenv("RL_SDN_BETA", "0.2").strip()),  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
        # Specify, which exploration sub-type to use (usually, the algo's "default"
        # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
        "sub_exploration": {
            "type": "StochasticSampling",
        }
    }

if string_to_bool(os.getenv("RL_SDN_AUTOREG", False)):
    ModelCatalog.register_custom_model(
        "autoregressive_model",
        AutoregressiveActionModel,
    )
    ModelCatalog.register_custom_action_dist(
        "binary_autoreg_dist",
        BinaryAutoregressiveDistribution,
    )

    print("Using custom mode and custom action distro")

    config["multiagent"]["policies"]["defender"] = PolicySpec(config={
                    "agent_id": 1,
                    "model": {
                        "custom_model": "autoregressive_model",
                        "custom_action_dist": "binary_autoreg_dist",
                        "use_lstm":  string_to_bool(os.getenv("RL_SDN_DEFENDERLSTM", False)),
                        "vf_share_layers": False
                    },
                    "framework": "tf",
                })

if os.getenv("RL_SDN_TRAINER", "PPO").lower() == "ppo":
    config["entropy_coeff"] = float(os.getenv("RL_SDN_ENTROPYCOEFF", "0.1").strip())
    config["kl_coeff"] = float(os.getenv("RL_SDN_KLCOEFF", "0.3").strip())