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


class CustomMetricsCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.actions = {"attacker": [], "defender": []}

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
        print('episode = {}'.format(episode.episode_id))

        #print("on_postprocess_traj info = {}".format(info))
        #print("on_postprocess_traj sample_obj = {}".format(sample_obj))
        #print('actions = {}'.format(postprocessed_batch.columns(["actions"])))

        self.actions[agent_id].append(postprocessed_batch.columns(["actions"]))
