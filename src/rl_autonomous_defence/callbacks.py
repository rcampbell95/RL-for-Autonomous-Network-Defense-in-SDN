"""
Custom metric to record winner for win percentage
"""

from typing import Dict
import math
import numpy as np

import ray
from ray import tune
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks



class SelfPlayCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.actions = {"attacker": [], "defender": []}
        # 0=RandomPolicy, 1=1st main policy snapshot,
        # 2=2nd main policy snapshot, etc..

        self.current_opponent = 0

        self.opponents = {"attacker": 0, "defender": 0}

    def _update_opponent(self, agent, trainer, result):
        opponent_reward = result["hist_stats"].pop(f"policy_{agent}_reward")


        won = 0
        for reward in opponent_reward:
            if reward > 0:
                won += 1
        win_rate = won / len(opponent_reward)
        result["win_rate"] = win_rate
        win_rate_threshold = 0.3
        print(f"Iter={trainer.iteration} win-rate={win_rate} -> ", end="")
        # If win rate is good -> Snapshot current policy and play against
        # it next, keeping the snapshot fixed and only improving the "main"
        # policy.
        if win_rate > win_rate_threshold:
            self.opponents[agent] += 1
            new_pol_id = f"{agent}_v{self.opponents[agent]}"
            print(f"adding new opponent to the mix ({new_pol_id}).")

            # Re-define the mapping function, such that "main" is forced
            # to play against any of the previously played policies
            # (excluding "random").
            def policy_mapping_fn(agent_id, episode, worker, **kwargs):
                # agent_id = [0|1] -> policy depends on episode ID
                # This way, we make sure that both policies sometimes play
                # (start player) and sometimes agent1 (player to move 2nd).
                if (episode.episode_id % 2) == 0:
                    if agent_id == "attacker":
                        if self.opponents["attacker"] == 0:
                            return "attacker"
                        return "attacker_v{}".format(
                            np.random.choice(list(range(1, self.opponents["attacker"] + 1)))
                        )
                    elif agent_id == "defender":
                        return "defender"
                else:
                    if agent_id == "attacker":
                        return "attacker"
                    elif agent_id == "defender":
                        if self.opponents["defender"] == 0:
                            return "defender"
                        return "defender_v{}".format(
                            np.random.choice(list(range(1, self.opponents["defender"] + 1)))
                        )

                print(agent_id)
                print(episode.episode_id)


            new_policy = trainer.add_policy(
                policy_id=new_pol_id,
                policy_cls=type(trainer.get_policy(agent)),
                policy_mapping_fn=policy_mapping_fn
            )

            # Set the weights of the new policy to the main policy.
            # We'll keep training the main policy, whereas `new_pol_id` will
            # remain fixed.
            opponent_state = trainer.get_policy(agent).get_state()
            new_policy.set_state(opponent_state)
            # We need to sync the just copied local weights (from main policy)
            # to all the remote workers as well.
            trainer.workers.sync_weights()


    def on_train_result(self, *, trainer, result, **kwargs):
        # Get the win rate for the train batch.
        # Note that normally, one should set up a proper evaluation config,
        # such that evaluation always happens on the already updated policy,
        # instead of on the already used train_batch.
        attacker_reward = result["hist_stats"].pop("policy_attacker_reward")

        # update defender snapshot
        defender_state = trainer.get_policy("defender").get_state()
        defender_snapshot = trainer.get_policy("defender_snapshot")
        defender_snapshot.set_state(defender_state)
        # We need to sync the just copied local weights (from main policy)
        # to all the remote workers as well.
        
        won = 0
        for reward in attacker_reward:
            if reward > 0:
                won += 1
        win_rate = won / len(attacker_reward)
        result["win_rate"] = win_rate
        win_rate_threshold = 0.3
        print(f"Iter={trainer.iteration} win-rate={win_rate} -> ", end="")
        # If win rate is good -> Snapshot current policy and play against
        # it next, keeping the snapshot fixed and only improving the "main"
        # policy.
        if win_rate > win_rate_threshold:
            self.current_opponent += 1
            new_pol_id = f"attacker_v{self.current_opponent}"
            print(f"adding new opponent to the mix ({new_pol_id}).")

            # Re-define the mapping function, such that "main" is forced
            # to play against any of the previously played policies
            # (excluding "random").
            def policy_mapping_fn(agent_id, episode, worker, **kwargs):
                # agent_id = [0|1] -> policy depends on episode ID
                # This way, we make sure that both policies sometimes play
                # (start player) and sometimes agent1 (player to move 2nd).
                if (episode.episode_id % 2) == 0:
                    if agent_id == "attacker":
                        return "attacker_v{}".format(
                            np.random.choice(list(range(1, self.current_opponent + 1)))
                        )
                    elif agent_id == "defender":
                        return "defender"
                else:
                    if agent_id == "attacker":
                        return "attacker"
                    elif agent_id == "defender":
                        return "defender_snapshot"

                print(agent_id)
                print(episode.episode_id)


            new_policy = trainer.add_policy(
                policy_id=new_pol_id,
                policy_cls=type(trainer.get_policy("attacker")),
                policy_mapping_fn=policy_mapping_fn
            )

            # Set the weights of the new policy to the main policy.
            # We'll keep training the main policy, whereas `new_pol_id` will
            # remain fixed.
            attacker_state = trainer.get_policy("attacker").get_state()
            new_policy.set_state(attacker_state)
            # We need to sync the just copied local weights (from main policy)
            # to all the remote workers as well.
            trainer.workers.sync_weights()
        else:
            print("not good enough; will keep learning ...")

        # +2 = main + random
        result["league_size"] = self.current_opponent + 1


    def check_winner(self, episode):
        attacker_reward = episode.last_reward_for("attacker")
        defender_reward = episode.last_reward_for("defender")

        try:
            assert math.floor(abs(attacker_reward)) == math.floor(abs(defender_reward))
        except AssertionError:
            raise AssertionError(f"Attacker reward: {attacker_reward} - Defender reward - {defender_reward}")

        if attacker_reward > 0:
            return 1
        elif defender_reward > 0:
            return -1
        elif attacker_reward == 0 and defender_reward == 0:
            return 0
  
    def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, **kwargs):
        print("episode {} started".format(episode.episode_id))
        episode.hist_data["episode_winners"] = []


    def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, **kwargs):

        pass


    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       **kwargs):
        
        winner = self.check_winner(episode)

        print("episode {} ended with winner {}".format(episode.episode_id, winner))
        episode.custom_metrics["episode_winner"] = winner
        episode.hist_data["episode_winners"].append(winner)


    def on_postprocess_trajectory(
            self, worker: RolloutWorker, episode: MultiAgentEpisode,
            agent_id: str, policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):
        #print("postprocessed {} steps".format(postprocessed_batch.count))
        if "num_batches" not in episode.custom_metrics:
            episode.custom_metrics["num_batches"] = 0
        episode.custom_metrics["num_batches"] += 1 

        #if agent_id == 'attacker':
        print('agent_id = {}'.format(agent_id))
        print(f'policy_id = {policy_id}')
        print('episode = {}'.format(episode.episode_id))

        #print("on_postprocess_traj info = {}".format(info))
        #print("on_postprocess_traj sample_obj = {}".format(sample_obj))
        #print('actions = {}'.format(postprocessed_batch.columns(["actions"])))

        self.actions[agent_id].append(postprocessed_batch.columns(["actions"]))
