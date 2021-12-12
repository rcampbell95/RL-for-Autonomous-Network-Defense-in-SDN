"""
Custom metric to record winner for win percentage
"""

from typing import Dict
import argparse
import numpy as np

import ray
from ray import tune
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks


class CustomMetricsCallback(DefaultCallbacks):
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

        winner = None
        for key, value in episode.agent_rewards.items():
            print(episode.agent_rewards)
            if value == 1:
                winner = 0 if key[0] == "attacker" else 1
        if winner == None:
            winner = -1

        print("episode {} ended with winner {}".format(episode.episode_id, winner))
        episode_end_mapping = {"tie": -1, "attacker": 0, "defender": 1}
        episode.custom_metrics["episode_winner"] = winner
        episode.hist_data["episode_winners"].append(winner)


    def on_postprocess_trajectory(
            self, worker: RolloutWorker, episode: MultiAgentEpisode,
            agent_id: str, policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):
        print("postprocessed {} steps".format(postprocessed_batch.count))
        if "num_batches" not in episode.custom_metrics:
            episode.custom_metrics["num_batches"] = 0
        episode.custom_metrics["num_batches"] += 1
