from callbacks import SelfPlayCallback
from ray.rllib.policy.policy import PolicySpec
from action_mask_model import ActionMaskModel
from ray.rllib.agents.dqn import dqn
import os
from random_policy import RandomPolicy
from ray.rllib.models import ModelCatalog

from utils import select_policy, string_to_bool
from AutoregressiveActionsModel import BinaryAutoregressiveDistribution, AutoregressiveActionModel


#config["partial_obs"] = {"lstm": tune.grid_search([True, False])}


#if string_to_bool(os.getenv("RL_SDN_RANDOMOPPONENT", "False").strip()):
#    attacker_policy = PolicySpec(policy_class=RandomPolicy)
#else:

ModelCatalog.register_custom_model(
    "action_mask_model",
    ActionMaskModel,
)

attacker_config = {
                    "model": {
                        "use_lstm": string_to_bool(os.getenv("RL_SDN_ATTACKERLSTM", False)),
                        "vf_share_layers": False
                        },
                    #"vf_loss_coeff": [0.01],
                    "entropy_coeff": float(os.getenv("RL_SDN_ENTROPYCOEFF", "0.001").strip()),
                    #"kl_coeff": float(os.getenv("RL_SDN_KLCOEFF", "0.3").strip()),
                    #"num_sgd_iter": float(os.getenv("RL_SDN_SGDITER", 30)),
                    #"vf_clip_param": float(os.getenv("RL_SDN_VFCLIP", 10)),
                    #"lambda": float(os.getenv("RL_SDN_LAMBDA", 1)),
                    #"gamma": float(os.getenv("RL_SDN_GAMMA", 0.99)),
                    "clip_param": float(os.getenv("RL_SDN_CLIP", 0.3)),
                }

defender_config = {
                    "model": {
                        "use_lstm":  string_to_bool(os.getenv("RL_SDN_DEFENDERLSTM", False)),
                        "vf_share_layers": False
                    },
                    #"vf_loss_coeff": [0.01],
                    "entropy_coeff": float(os.getenv("RL_SDN_ENTROPYCOEFF", "0.001").strip()),
                    #"kl_coeff": float(os.getenv("RL_SDN_KLCOEFF", "0.3").strip()),
                    #"num_sgd_iter": float(os.getenv("RL_SDN_SGDITER", 30)),
                    #"vf_clip_param": float(os.getenv("RL_SDN_VFCLIP", 10)),
                    #"lambda": float(os.getenv("RL_SDN_LAMBDA", 1)),
                    #"gamma": float(os.getenv("RL_SDN_GAMMA", 0.99)),
                    "clip_param": float(os.getenv("RL_SDN_CLIP", 0.3)),

                }

if string_to_bool(os.getenv("RL_SDN_MASKEDACTIONS", False)):
    defender_config["model"]["custom_model"]  =  "action_mask_model"
    attacker_config["model"]["custom_model"]  =  "action_mask_model"



if os.getenv("RL_SDN_ACTIONSPACE", "multi") == "autoreg":
    ModelCatalog.register_custom_model(
        "autoregressive_model",
        AutoregressiveActionModel,
    )
    ModelCatalog.register_custom_action_dist(
        "binary_autoreg_dist",
        BinaryAutoregressiveDistribution,
    )

    print("Using custom mode and custom action distro")

    defender_config["model"]["custom_model"] = "autoregressive_model"
    defender_config["model"]["custom_action_dist"] = "binary_autoreg_dist"


config = {
    "env": "AutonomousDefenceEnv",
    "callbacks": SelfPlayCallback,
    "log_level": "DEBUG",
    "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
    "rollout_fragment_length": 200,
    "train_batch_size": os.getenv("RL_SDN_BATCHSIZE", 1000),
    "num_workers": 2,
    "num_envs_per_worker": 4,
    "num_cpus_for_driver": 1,
    "lr": float(os.getenv("RL_SDN_LR", 5e-4)),
    "lr_schedule": [
        [0, float(os.getenv("RL_SDN_LR", 5e-4))],
        [400000, 1e-5]
    ],
    #"min_train_timesteps_per_reporting": 5000,
    "timesteps_per_iteration": 2000,
    "framework": "tf",
    #"eager_tracing": True,
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
            "CPU": 2
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
                "attacker": PolicySpec(config=attacker_config),
                "defender_v0": PolicySpec(policy_class=RandomPolicy), #PolicySpec(config=defender_config),
                "attacker_v0": PolicySpec(policy_class=RandomPolicy), #PolicySpec(config=attacker_config),
                "defender": PolicySpec(config=defender_config),
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