import os


def string_to_bool(bool_string: str) -> bool:
    if isinstance(bool_string, bool):
        return bool_string

    if bool_string.lower() == "false":
        return False
    elif bool_string.lower() == "true":
        return True


train_config = {
    "experiment_dir": os.getenv("RL_SDN_EXPERIMENT_DIRECTORY"),
    "train": {
        "horizon": int(os.environ.get("RL_SDN_HORIZON", 200)),
        "train_batch_size": os.getenv("RL_SDN_BATCHSIZE", 1000),
        "num_gpus": float(os.environ.get("RLLIB_NUM-GPUS", "1")),
        "lr": float(os.getenv("RL_SDN_LR", 3e-4))
    },
    "environment": {
        "env_name": os.getenv("RL_SDN_ENVIRONMENT", "AutonomousDefenceEnv").strip(),
        "timesteps": int(os.getenv("RL_SDN_TIMESTEPS", "4000000").strip()),
        "topology": os.getenv("RL_SDN_TOPOLOGY", "random").strip(),
        "reward": os.getenv("RL_SDN_REWARD", "reward-scaled").strip(),
        "mean_impact_score": float(os.getenv("RL_SDN_MIS", 4.311829)),
        "mean_exploitability_score": float(os.getenv("RL_SDN_MES", 2.592744)),
        "std_exploitability_score": float(os.getenv("RL_SDN_STDES", "0.01").strip()),
        "std_impact_score": float(os.getenv("RL_SDN_STDIS", "0.01").strip()),
        "network_size": int(os.getenv("RL_SDN_NETWORKSIZE", 16)),
        "network_size_max": int(os.environ.get("RL_SDN_NETWORKSIZE-MAX", os.environ.get("RL_SDN_NETWORKSIZE", 16))),
    },
    "self-play": {
        "win_threshold": float(os.getenv("RL_SDN_WINTHRESH", "0").strip()),
        "recent_agents_probs": float(os.getenv("RL_SDN_RECENT-AGENT-PROBS", 0.8)),
        "top_k_percent": float(os.getenv("RL_SDN_TOP-K", 0.1)),
        "snapshot_frequency": float(os.getenv("RL_SDN_SNAPSHOT-FREQ", 10))
    },
    "agent": {
        "action_space": os.getenv("RL_SDN_ACTIONSPACE", "multi").strip(),
        "masked_actions": os.getenv("RL_SDN_MASKEDACTIONS", True),
        #"attacker" : {
        #    "vf_loss_coeff": float(os.getenv("RL_SDN_VF-COEFF", 1)),
        #    "entropy_coeff": ,
        #    "kl_coeff": ,
        #    "gamma": ,
        #    "num_iter_sgd": ,
        #    "custom_model_config": {}
        #},
        #"defender": {
        #    "vf_loss_coeff": float(os.getenv("RL_SDN_VF-COEFF", 1)),S
        #},
        "debug": {
            "attacker": {
                "cheat_agent": string_to_bool(os.getenv("RL_SDN_ATTACKER-CHEAT", "False").strip())
            },
            "defender": {
                "cheat_agent": string_to_bool(os.getenv("RL_SDN_DEFENDER-CHEAT", "False").strip())
            }
        }
    },
    "actions" : {
        "attacker": {
            "explore_topo": string_to_bool(os.getenv("RL_SDN_EXPLORETOPO", "True").strip()),
            "scan_vuln": string_to_bool(os.getenv("RL_SDN_SCANVULN", "True").strip()),
            "attack_vuln": string_to_bool(os.getenv("RL_SDN_ATTACKVULN", "True").strip())
        },
        "defender": {
            "check_status": string_to_bool(os.getenv("RL_SDN_CHECKSTATUS", "True").strip()),
            "isolate_node": string_to_bool(os.getenv("RL_SDN_ISOLATENODE", "True").strip()),
            "move_flag": string_to_bool(os.getenv("RL_SDN_MOVEFLAG", "True").strip())
        }
    }
}

