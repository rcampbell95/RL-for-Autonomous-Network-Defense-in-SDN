"""
Custom metric to record winner for win percentage
"""
import os
from typing import Dict
import copy
import math
from collections import defaultdict

import numpy as np

from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray import tune
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks

from rl_autonomous_defence.train_config import train_config

from rl_autonomous_defence.agent_config import (
    ATTACKER_CONFIG,
    DEFENDER_CONFIG
)

from rl_autonomous_defence import utils


class SelfPlayCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.agents = ["attacker", "defender"]
        self.actions = defaultdict(list)

        self.current_opponent = 0
        self.attacker_policies = []
        self.defender_policies = []

        self.elo_scores = {"attacker": 500,
                           "attacker_v0": 500,
                           "defender": 500,
                           "defender_v0": 500}

        self.win_rate_start = train_config["self-play"]["win_threshold"]

        self.opponents = defaultdict(int) #{agent: 0 for agent in self.agents}
        self.win_rate_thresholds = {agent: self.win_rate_start for agent in self.agents}

        self.recent_agents_probs = train_config["self-play"]["recent_agents_probs"]
        self.top_k_percent = train_config["self-play"]["top_k_percent"]

        self.pool_decay_rate = 1 / (train_config["environment"]["timesteps"] // train_config["train"]["horizon"])
        self.environment_randomness = 1 / (train_config["environment"]["timesteps"] // train_config["train"]["horizon"])

        self.opponent_snapshot_freq = train_config["self-play"]["snapshot_frequency"]
        self.rng = np.random.default_rng()
        self.opponent_pool_probs = {agent: [1] for agent in self.agents}

    def _update_opponent(self, agent, trainer, result):
        try:
            win_count = result["hist_stats"].pop(f"{agent}_win_count")
        except Exception as e:
            print(result["hist_stats"])
            raise e
        won = 0
        for win in win_count:
            if win > 0:
                won += 1
        win_rate = won / len(win_count) if len(win_count) > 0 else 0
        
        result[f"{agent}_pool_win_rate"] = win_rate

        #ste
        # ps_since_update = result["num_env_steps_sampled"] - self.last_added[agent] 

        print(f"Iter={trainer.iteration} win-rate={win_rate} -> ", end="")
        # If win rate is good -> Snapshot current policy and play against
        # it next, keeping the snapshot fixed and only improving the "main"
        # policy.
        #if win_rate > self.win_rate_thresholds[agent]:
        if (trainer.iteration % self.opponent_snapshot_freq) == 0:
            self.opponents[agent] += 1
            self.win_rate_thresholds[agent] += 0.025
            new_pol_id = f"{agent}_v{self.opponents[agent]}"
            #self.last_added[agent] = result["num_env_steps_sampled"]
            print(f"adding new opponent to the mix ({new_pol_id}).")
            self.elo_scores[new_pol_id] = self.elo_scores[agent]

            self.opponent_pool_probs[agent] = self.adjust_policy_probs(agent)

            try:
                assert len(self.opponent_pool_probs[agent]) == (self.opponents[agent] + 1)
            except Exception as e:
                raise e

            # Re-define the mapping function, such that "main" is forced
            # to play against any of the previously played policies
            # (excluding "random").

            def policy_mapping_fn(agent_id, episode, worker, **kwargs):
                if (episode.episode_id % 2) == 0:
                    if agent_id == "attacker":
                        agents = list(range(0, self.opponents[agent_id] + 1))
                        agent_selection = self.rng.choice(agents, 1, p=self.opponent_pool_probs[agent_id]).item()
                        return f"{agent_id}_v{agent_selection}"
                    elif agent_id == "defender":
                        return "defender"
                else:
                    if agent_id == "attacker":
                        return "attacker"
                    elif agent_id == "defender":
                        #if self.opponents["defender"] == 0:
                        #    return "defender"
                        agents = list(range(0, self.opponents[agent_id] + 1))
                        agent_selection = self.rng.choice(agents, 1, p=self.opponent_pool_probs[agent_id]).item()
                        return f"{agent_id}_v{agent_selection}"

            config = copy.deepcopy(trainer.get_policy(agent).config)
            config["model"]["custom_model_config"]["masked_actions"] = False

            new_policy = trainer.add_policy(
                policy_id=new_pol_id,
                policy_cls=type(trainer.get_policy(agent)),
                config=config,
                policy_mapping_fn=policy_mapping_fn,
                action_space=trainer.get_policy(agent).action_space,
                policies_to_train=["attacker", "defender"]
            )
            # Set the weights of the new policy to the main policy.
            # We'll keep training the main policy, whereas `new_pol_id` will
            # remain fixed.
            opponent_state = trainer.get_policy(agent).get_state()
            try:
                new_policy.set_state(opponent_state)
                past_opponent = trainer.get_policy(f"{agent}_v{self.opponents[agent] - 1}")
                saved_weights = [trainer.get_policy(agent).get_weights(),
                                 past_opponent.get_weights()]
                mean_weights = []
                for weights_list_tuple in zip(*saved_weights): 
                    mean_weights.append(
                        np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)]))
                new_policy.set_weights(mean_weights)
            except AssertionError as e:
                opponent_weights = opponent_state["weights"]
                agent_config = config["model"]
                raise Exception(f"Error setting opponent weights. \n\
                                Agent: {agent} \n\
                                Agent policy: {type(trainer.get_policy(agent))} \n\
                                Agent model: {trainer.get_policy(agent).model} \n\
                                New policy model: {new_policy.model}")
            # We need to sync the just copied local weights (from main policy)
            # to all the remote workers as well.
            trainer.workers.sync_weights()
        else:
            print("not good enough; will keep learning ...")

        result[f"league_size_{agent}"] = self.opponents[agent] + 1


    def adjust_policy_probs(self, agent_id):
        agents = list(range(0, self.opponents[agent_id] + 1))
        recent_agents_probs = max(0.1, self.recent_agents_probs - self.pool_decay_rate)
        #recent_agents_probs = recent_agents_probs
        past_agents_probs = min(0.9, 1.0 - recent_agents_probs)

        recent_agents_pool_size = math.ceil(len(agents) * self.top_k_percent)
        past_agents_pool_size = math.floor(len(agents) * (1 - self.top_k_percent))

        if past_agents_pool_size > 0:
            past_probs = [past_agents_probs / past_agents_pool_size] * past_agents_pool_size
        else:
            past_probs = []
        recent_probs = [recent_agents_probs / recent_agents_pool_size] * recent_agents_pool_size

        self.recent_agents_probs = recent_agents_probs

        try:
            assert (1 - (past_agents_probs + recent_agents_probs)) < 0.0001
        except AssertionError:
            raise AssertionError(f"Past agents probs {past_agents_probs} recent agent probs {recent_agents_probs}")

        return past_probs + recent_probs


    def on_train_result(self, *, trainer, result, **kwargs):
        # F: the win rate for the train batch.
        # Note that normally, one should set up a proper evaluation config,
        # such that evaluation always happens on the already updated policy,
        # instead of on the already used train_batch.

        # We need to sync the just copied local weights (from main policy)
        # to all the remote workers as well.
        if "attacker_win_count" in result["hist_stats"]:
            self._update_opponent("attacker", trainer, result)
        if "defender_win_count" in result["hist_stats"]:
            self._update_opponent("defender", trainer, result)


    def check_winner(self, env, episode):
        if env.winner == "draw":
            winner = "draw"
        elif env.winner != "draw":
            winner = episode.policy_for(env.winner)
            trainable_has_won = 1 if winner in self.agents else 0

            print(f"Winner policy: {winner}")

            agent_id = "attacker" if "attacker" in winner else "defender"
            if trainable_has_won:
                trainable = winner
            elif not trainable_has_won:
                trainable = self.agents[self.agents.index(agent_id) - 1]
            

            # if trainable_has_won:
            attacker_policy = episode.policy_for("attacker")
            defender_policy = episode.policy_for("defender")

            episode.custom_metrics[f"{trainable}_win_count"] = trainable_has_won
            episode.hist_data[f"{trainable}_win_count"].append(trainable_has_won)
            #else:
            #    agent_id = "attacker" if "attacker" in winner else "defender"
            #    loser = self.agents[self.agents.index(agent_id) - 1]

            #    episode.custom_metrics[f"{loser}_win_count"] = trainable_has_won
            #    episode.hist_data[f"{loser}_win_count"].append(trainable_has_won)

        if "attacker" in winner:
            return attacker_policy
        elif "defender" in winner:
            return defender_policy
        elif "draw" in winner:
            return "tie"


    def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, **kwargs):
        print("episode {} started".format(episode.episode_id))

        for agent in ["attacker", "defender"]:
            episode.hist_data[f"{agent}_num_compromised_nodes"] = []
            episode.hist_data[f"{agent}_wins"] = []
            episode.hist_data[f"{agent}_ties"] = []
            episode.hist_data[f"episode_len_{agent}"] = []
            episode.hist_data[f"{agent}_actions"] = []

            episode.hist_data[f"{agent}_total_impact"] = []
            episode.hist_data[f"{agent}_total_cost"] = []

            episode.hist_data[f"{agent}_win_count"] = []

        attacker_policy = episode.policy_for("attacker")
        defender_policy = episode.policy_for("defender")

        episode.hist_data[f"{attacker_policy}_elo_scores"] = []
        episode.hist_data[f"{defender_policy}_elo_scores"] = []

        episode.hist_data["episode_winners"] = []

        episode.custom_metrics["recent_agents_probs"] = self.recent_agents_probs

        for agent in ["attacker", "defender"]:
            self.opponent_pool_probs[agent] = self.adjust_policy_probs(agent)


    def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, **kwargs):

        pass


    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       **kwargs):
        unwrapped_env = base_env.get_unwrapped()[0].env.unwrapped

        winner = self.check_winner(unwrapped_env, episode)

        attacker_policy = episode.policy_for("attacker") or "attacker"
        defender_policy = episode.policy_for("defender") or "defender"
        agent = "attacker" if attacker_policy == "attacker" else "defender"
        
        episode.custom_metrics[f"{agent}_win"] = 1 if winner == agent else 0
        episode.hist_data[f"{agent}_wins"].append(episode.custom_metrics[f"{agent}_win"])

        if winner == agent:
            episode.custom_metrics[f"{agent}_total_impact"] = unwrapped_env.total_impact 
            episode.hist_data[f"{agent}_total_impact"].append(episode.custom_metrics[f"{agent}_total_impact"])

            episode.custom_metrics[f"{agent}_total_cost"] = unwrapped_env.total_cost
            episode.hist_data[f"{agent}_total_cost"].append(episode.custom_metrics[f"{agent}_total_cost"])

            episode.custom_metrics[f"{agent}_num_compromised_nodes"] = unwrapped_env.num_compromised_nodes
            episode.hist_data[f"{agent}_num_compromised_nodes"].append(episode.custom_metrics[f"{agent}_num_compromised_nodes"])

        episode.custom_metrics[f"{agent}_tie"] = 1 if winner == "tie" else 0
        episode.hist_data[f"{agent}_ties"].append(episode.custom_metrics[f"{agent}_tie"])
        try:
            episode.custom_metrics[f"episode_len_{agent}"] = episode.length
            episode.hist_data[f"episode_len_{agent}"].append(episode.custom_metrics[f"episode_len_{agent}"])
        except:
            print(episode.hist_data.keys())

        if winner:
            if attacker_policy not in self.elo_scores:
                self.elo_scores[attacker_policy] = self.elo_scores["attacker"] 
            if defender_policy not in self.elo_scores:
                self.elo_scores[defender_policy] = self.elo_scores["defender"]
            attacker_rating = utils.elo_score(self.elo_scores[attacker_policy],
                                                self.elo_scores[defender_policy],
                                                100,
                                                winner in attacker_policy)
            defender_rating = utils.elo_score(self.elo_scores[defender_policy],
                                            self.elo_scores[attacker_policy],
                                            100,
                                            winner in defender_policy)

        episode.custom_metrics[f"{attacker_policy}_elo_score"] = attacker_rating
        episode.custom_metrics[f"{defender_policy}_elo_score"] = defender_rating
        episode.hist_data[f"{attacker_policy}_elo_scores"].append(attacker_rating)
        episode.hist_data[f"{defender_policy}_elo_scores"].append(defender_rating)

        stdis_current = float(os.getenv("RL_SDN_STDIS", "0.01").strip()) #train_config["environment"]["std_impact_score"]
        stdes_current = float(os.getenv("RL_SDN_STDES", "0.01").strip()) #train_config["environment"]["std_exploitability_score"]

        network_size_current = train_config["environment"]["network_size"] 
        network_size_max = train_config["environment"]["network_size_max"] 
        
        update_network_size = min(network_size_max, network_size_current + .1)
        # Creates an issue where network size never changes
        os.environ["RL_SDN_NETWORKSIZE"] = str(update_network_size)
        
        episode.custom_metrics["curr_network_size"] = int(update_network_size)

        os.environ["RL_SDN_STDIS"] = str(min(1, stdis_current + self.environment_randomness))
        os.environ["RL_SDN_STDES"] = str(min(1, stdes_current + self.environment_randomness))

        episode.custom_metrics["impact_score_std"] = float(os.environ["RL_SDN_STDIS"])
        episode.custom_metrics["exploitability_score_std"] = float(os.environ["RL_SDN_STDES"])

    def on_learn_batch(self, policy,
                       train_batch: SampleBatch, result: dict):
        pass
        #os.environ["RL_SDN_HORIZON"] = str(min(int(os.environ["RL_SDN_HORIZON"]) + 10, 200))
        #rewards = train_batch["rewards"]

        #normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        #train_batch["rewards"] = normalized_rewards

        

    def on_postprocess_trajectory(
            self, worker: RolloutWorker, episode: MultiAgentEpisode,
            agent_id: str, policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):

        if "num_batches" not in episode.custom_metrics:
            episode.custom_metrics["num_batches"] = 0
        episode.custom_metrics["num_batches"] += 1 

        self.actions[agent_id].append(postprocessed_batch.columns(["actions"]))
