import os

from ray.tune.registry import register_env
# import the pettingzoo environment
# import rllib pettingzoo interface
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.agents.registry import get_trainer_class
import ray
from ray import tune

from rl_autonomous_defence import ade
from rl_autonomous_defence import policy_config
from rl_autonomous_defence.CheckpointWrapper import CheckpointWrapperPPO



if __name__ == "__main__":
    ray.init(log_to_driver=False,
             num_cpus=4)

    register_env('AutonomousDefenceEnv', lambda x: ParallelPettingZooEnv(ade.env()))

    stop = {
        #"episodes_total": 150,
        "timesteps_total": int(os.getenv("RL_SDN_TIMESTEPS", "45000").strip())
        #"episode_reward_mean": 1000,
    }

    #algo = 
    #trainer = get_trainer_class(algo)()

    #config = ppo.DEFAULT_CONFIG
    #config.update(policy_config.config)

    results = tune.run(CheckpointWrapperPPO,
                        config=policy_config.config, 
                        stop=stop, 
                        verbose=1,
                        checkpoint_freq=1,
                        checkpoint_at_end=True,
                        local_dir=os.getenv("RL_SDN_EXPERIMENT_DIRECTORY").strip())
                        #num_samples=10)


    ray.shutdown()