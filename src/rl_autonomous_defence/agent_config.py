import os

from ray.rllib.models import ModelCatalog

from rl_autonomous_defence.fcn_attacker_action_mask_model import FCNAttackerActionMaskModel
from rl_autonomous_defence.gcn_attacker_action_mask_model import GCNAttackerActionMaskModel

from rl_autonomous_defence.fcn_defender_action_mask_model import FCNDefenderActionMaskModel
from rl_autonomous_defence.gcn_defender_action_mask_model import GCNDefenderActionMaskModel

from rl_autonomous_defence.attacker_autoregressive_distribution import AttackerAutoregressiveDistribution
from rl_autonomous_defence.defender_autoregressive_distribution import DefenderAutoregressiveDistribution
from rl_autonomous_defence.autoregressive_model import AutoregressiveActionModel
from rl_autonomous_defence import ade
from rl_autonomous_defence.utils import string_to_bool


ModelCatalog.register_custom_model(
    "fcn_attacker_action_mask_model",
    FCNAttackerActionMaskModel,
)

ModelCatalog.register_custom_model(
    "gcn_attacker_action_mask_model",
    GCNAttackerActionMaskModel,
)


ModelCatalog.register_custom_model(
    "fcn_defender_action_mask_model",
    FCNDefenderActionMaskModel,
)

ModelCatalog.register_custom_model(
    "gcn_defender_action_mask_model",
    GCNDefenderActionMaskModel,
)

ATTACKER_CONFIG = {
                    "model": {
                        "use_attention": string_to_bool(os.getenv("RL_SDN_ATTACK-ATTENTION", False)),
                        "vf_share_layers": False,
                        "conv_filters": [[4, 3, 2], [8, 3, 2], [11, 2, 2]],
                        "conv_activation": os.getenv("RL_SDN_CNN-ACTIVATION", "relu"),
                        "post_fcnet_hiddens": [256, 256],
                        "post_fcnet_activation": os.getenv("RL_SDN_FCNET-ACTIVATION", "relu"),
                        "custom_model_config": {
                            "masked_actions": os.getenv("RL_SDN_MASKEDACTIONS", False)
                        }
                    },
                    "clip_param": float(os.getenv("RL_SDN_CLIP", 0.2)),
                    "vf_loss_coeff": 0.1,
                    "entropy_coeff": 0.1,
                    "gamma": float(os.getenv("RL_SDN_GAMMA", 0.995)),
                }

DEFENDER_CONFIG = {
                    "model": {
                        "use_attention":  string_to_bool(os.getenv("RL_SDN_DEFEND-ATTENTION", False)),
                        "vf_share_layers": False,
                        "conv_filters": [[4, 2, 2], [8, 2, 2], [11, 2, 2]],
                        "conv_activation": os.getenv("RL_SDN_CNN-ACTIVATION", "relu"),
                        "post_fcnet_hiddens": [256, 256],
                        "post_fcnet_activation": os.getenv("RL_SDN_FCNET-ACTIVATION", "relu"),
                        "custom_model_config": {
                            "masked_actions": os.getenv("RL_SDN_MASKEDACTIONS", False)
                        }
                    },
                    "clip_param": float(os.getenv("RL_SDN_CLIP", 0.2)),
                    "vf_loss_coeff": 0.1,
                    "entropy_coeff": 0.1,
                    #"kl_coeff": 0,
                    #"kl_target": 10,
                    "gamma": float(os.getenv("RL_SDN_GAMMA", 0.995))
                }

if string_to_bool(os.getenv("RL_SDN_MASKEDACTIONS", False)):
    model_backbone = os.getenv("RL_SDN_MODEL-BACKBONE", "fcn").lower()

    assert model_backbone in ["fcn", "gcn"]

    DEFENDER_CONFIG["model"]["custom_model"] = f"{model_backbone}_defender_action_mask_model"
    ATTACKER_CONFIG["model"]["custom_model"] = f"{model_backbone}_attacker_action_mask_model"


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
