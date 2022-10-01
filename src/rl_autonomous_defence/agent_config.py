import os

from ray.rllib.models import ModelCatalog

from rl_autonomous_defence.attacker_action_mask_model import AttackerActionMaskModel
from rl_autonomous_defence.defender_action_mask_model import DefenderActionMaskModel
from rl_autonomous_defence.attacker_autoregressive_distribution import AttackerAutoregressiveDistribution
from rl_autonomous_defence.defender_autoregressive_distribution import DefenderAutoregressiveDistribution
from rl_autonomous_defence.autoregressive_model import AutoregressiveActionModel
from rl_autonomous_defence import ade
from rl_autonomous_defence.utils import string_to_bool


ModelCatalog.register_custom_model(
    "attacker_action_mask_model",
    AttackerActionMaskModel,
)
ModelCatalog.register_custom_model(
    "defender_action_mask_model",
    DefenderActionMaskModel,
)

ATTACKER_CONFIG = {
                    "model": {
                        "use_attention": string_to_bool(os.getenv("RL_SDN_ATTACK-ATTENTION", True)),
                        "vf_share_layers": False,
                        "max_seq_len": 10
                        },
                    "clip_param": float(os.getenv("RL_SDN_CLIP", 0.2)),
                    "vf_loss_coeff": 0.1,
                    "entropy_coeff": 0.1,
                    "gamma": float(os.getenv("RL_SDN_GAMMA", 0.995))
                }

DEFENDER_CONFIG = {
                    "model": {
                        "use_attention":  string_to_bool(os.getenv("RL_SDN_DEFEND-ATTENTION", True)),
                        "vf_share_layers": False
                    },
                    "clip_param": float(os.getenv("RL_SDN_CLIP", 0.2)),
                    "vf_loss_coeff": 0.1,
                    "entropy_coeff": 0.01,
                    #"kl_coeff": 0,
                    #"kl_target": 10,
                    "gamma": float(os.getenv("RL_SDN_GAMMA", 0.995))                    
                }

if string_to_bool(os.getenv("RL_SDN_MASKEDACTIONS", False)):
    env = ade.env()
    DEFENDER_CONFIG["model"]["custom_model"]  = "defender_action_mask_model"
    ATTACKER_CONFIG["model"]["custom_model"]  = "attacker_action_mask_model"


if os.getenv("RL_SDN_ACTIONSPACE", "multi") == "autoreg":
    ModelCatalog.register_custom_model(
        "autoregressive_model",
        AutoregressiveActionModel,
    )
    ModelCatalog.register_custom_action_dist(
        "attacker_autoregressive_distribution",
        AttackerAutoregressiveDistribution,
    )
    ModelCatalog.register_custom_action_dist(
        "defender_autoregressive_distribution",
        DefenderAutoregressiveDistribution,
    )

    print("Using custom mode and custom action distro")

    DEFENDER_CONFIG["model"]["custom_model"] = "autoregressive_model"
    DEFENDER_CONFIG["model"]["custom_action_dist"] = "defender_autoregressive_distribution"

    ATTACKER_CONFIG["model"]["custom_model"] = "autoregressive_model"
    ATTACKER_CONFIG["model"]["custom_action_dist"] = "attacker_autoregressive_distribution"
