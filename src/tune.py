import ray.rllib.agents.pg as PG
#import ray.rllib.agents. as PG

from ray.tune.registry import register_env
# import the pettingzoo environment
# import rllib pettingzoo interface
from ray.rllib.env import PettingZooEnv
from ray.rllib.agents.pg import PGTrainer

from ray import tune
import ade
import policy_config
import os


if __name__ == "__main__":

    register_env('AutonomousDefenceEnv', lambda x: PettingZooEnv(ade.env()))

    stop = {
        #"episodes_total": 150,
        "timesteps_total": int(os.getenv("RL_SDN_TIMESTEPS", "45000").strip())
        #"episode_reward_mean": 1000,
    }

    algorithms = ["PPO"] #["A2C", "PG", "PPO"]

    #   params = {"attackerLSTM": "False", "defenderLSTM": "True"}
    #
    #  trial_name = "_".join([f"{key}={str(value)}" for key, value in params.items()])


    results = tune.run(os.getenv("RL_SDN_TRAINER", "PPO").strip(),
                        config=policy_config.config, 
                        stop=stop, 
                        verbose=1,
                        checkpoint_freq=1,
                        checkpoint_at_end=True,
                        local_dir=os.getenv("RL_SDN_EXPERIMENT_DIRECTORY").strip())
                        #num_samples=10)