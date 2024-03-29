import os
import json

from ray.tune.registry import register_env
# import the pettingzoo environment
# import rllib pettingzoo interface
from ray.rllib.env import PettingZooEnv
import ray
from ray import tune

from rl_autonomous_defence import ade
from rl_autonomous_defence import toy_env
from rl_autonomous_defence import policy_config
from rl_autonomous_defence.CheckpointWrapper import CheckpointWrapperPPO
from rl_autonomous_defence.train_config import train_config


if __name__ == "__main__":
    ray.init(log_to_driver=True, num_cpus=18)

    register_env('toy-env', lambda x: PettingZooEnv(toy_env.env()))
    register_env('AutonomousDefenceEnv', lambda x: PettingZooEnv(ade.env()))

    policy_config.config["env"] = "AutonomousDefenceEnv" 
    # train_config["environment"]["env_name"]

    #if policy_config.config["env"] == "toy-env":
    #    del policy_config.config["callbacks"]

    stop = {
        "timesteps_total": train_config["environment"]["timesteps"]
    }

    try:
        results = tune.run(CheckpointWrapperPPO,
                        config=policy_config.config, 
                        stop=stop, 
                        verbose=1,
                        checkpoint_freq=1,
                        checkpoint_at_end=True,
                        local_dir=train_config["experiment_dir"])
                        #num_samples=10)
    except Exception as e:
        raise(e)
    finally:
        ray.shutdown()


    config_output_file = os.path.join(train_config["experiment_dir"], "train_config.json")
    with open(config_output_file, "w") as fp:
        json.dump(train_config, fp)

