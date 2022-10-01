import os

train_config = {
    "experiment_dir": os.getenv("RL_SDN_EXPERIMENT_DIRECTORY"),
    "environment": {
        "timesteps": int(os.getenv("RL_SDN_TIMESTEPS", "45000").strip()),
        "topology": os.getenv("RL_SDN_TOPOLOGY", "clique").strip(),
        "reward": os.getenv("RL_SDN_REWARD", "gabirondo").strip(),
        "mean_impact_score": float(os.getenv("RL_SDN_MIS", 4.311829)),
        "mean_exploitability_score": float(os.getenv("RL_SDN_MES", 2.592744)),
        "std_exploitability_score": float(os.getenv("RL_SDN_STDES", "0.01").strip()),
        "std_impact_score": float(os.getenv("RL_SDN_STDIS", "0.01").strip()),
        "network_size": int(os.environ["RL_SDN_NETWORKSIZE"]),
        "network_size_max": int(os.environ.get("RL_SDN_NETWORKSIZE-MAX", os.environ["RL_SDN_NETWORKSIZE"])),
        "horizon": int(os.environ.get("RL_SDN_HORIZON", 200)),
    },
    "self-play": {
        "win_threshold": float(os.getenv("RL_SDN_WINTHRESH", "0").strip()),
        "recent_agents_probs": float(os.getenv("RL_SDN_RECENT-AGENT-PROBS", 0.8)),
        "top_k_percent": float(os.getenv("RL_SDN_TOP-K", 0.1)),
        "snapshot_frequency": float(os.getenv("RL_SDN_SNAPSHOT-FREQ", 10))
    },
    "agent": {
        "action_space": os.getenv("RL_SDN_ACTIONSPACE", "multi").strip()
    },
    "actions" : {
        "attacker": {
            "explore_topo": os.getenv("RL_SDN_EXPLORETOPO", "True").strip(),
            "scan_vuln": os.getenv("RL_SDN_SCANVULN", "True").strip(),
            "attack_vuln": os.getenv("RL_SDN_ATTACKVULN", "True").strip()
        },
        "defender": {
            "check_status": os.getenv("RL_SDN_CHECKSTATUS", "True").strip(),
            "isolate_node": os.getenv("RL_SDN_ISOLATENODE", "True").strip(),
            "move_flag": os.getenv("RL_SDN_MOVEFLAG", "True").strip()
        }
    }
}
       


