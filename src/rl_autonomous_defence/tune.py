import ray.rllib.agents.pg as ppo
#import ray.rllib.agents. as PG

from ray.tune.registry import register_env
# import the pettingzoo environment
# import rllib pettingzoo interface
from ray.rllib.env import PettingZooEnv
from ray.rllib.agents.registry import get_trainer_class
import ray

from ray import tune
import ade
import policy_config
import os


if __name__ == "__main__":
    ray.init(num_cpus=3)

    register_env('AutonomousDefenceEnv', lambda x: PettingZooEnv(ade.env()))

    stop = {
        #"episodes_total": 150,
        "timesteps_total": int(os.getenv("RL_SDN_TIMESTEPS", "45000").strip())
        #"episode_reward_mean": 1000,
    }

    #algo = 
    #trainer = get_trainer_class(algo)()

    config = ppo.DEFAULT_CONFIG
    config.update(policy_config.config)

    results = tune.run(os.getenv("RL_SDN_TRAINER", "PPO").strip(),
                        config=config, 
                        stop=stop, 
                        verbose=1,
                        checkpoint_freq=1,
                        checkpoint_at_end=True,
                        local_dir=os.getenv("RL_SDN_EXPERIMENT_DIRECTORY").strip())
                        #num_samples=10)