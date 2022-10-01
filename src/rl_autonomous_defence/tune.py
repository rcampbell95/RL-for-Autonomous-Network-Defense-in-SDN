from ray.tune.registry import register_env
# import the pettingzoo environment
# import rllib pettingzoo interface
from ray.rllib.env import PettingZooEnv
import ray
from ray import tune

from rl_autonomous_defence import ade
from rl_autonomous_defence import policy_config
from rl_autonomous_defence.CheckpointWrapper import CheckpointWrapperPPO
from rl_autonomous_defence.train_config import train_config


if __name__ == "__main__":
    ray.init(log_to_driver=False, num_cpus=4)

    register_env('AutonomousDefenceEnv', lambda x: PettingZooEnv(ade.env()))

    stop = {
        "timesteps_total": train_config["environment"]["timesteps"]
    }

    results = tune.run(CheckpointWrapperPPO,
                        config=policy_config.config, 
                        stop=stop, 
                        verbose=1,
                        checkpoint_freq=1,
                        checkpoint_at_end=True,
                        local_dir=train_config["experiment_dir"])
                        #num_samples=10)


    ray.shutdown()