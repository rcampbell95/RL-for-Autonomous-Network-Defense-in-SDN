"""Utilities for managing policy and environment."""
import math
import os

import numpy as np
from numpy.random import default_rng

import tensorflow as tf

from rl_autonomous_defence.train_config import train_config

NETWORK_SAMPLE_THRESHOLD = float(os.environ.get("RL_SDN_STDIS", "0.01"))


def select_policy(agent_id, episode, worker, **kwargs):
    if (episode.episode_id % 2) == 0:
        if agent_id == "attacker":
            return "attacker_v0"
        elif agent_id == "defender":
            return "defender"
    else:
        if agent_id == "attacker":
            return "attacker"
        elif agent_id == "defender":
            return "defender_v0"


def eval_policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return agent_id


def string_to_bool(bool_string: str) -> bool:
    if isinstance(bool_string, bool):
        return bool_string

    if bool_string.lower() == "false":
        return False
    elif bool_string.lower() == "true":
        return True


def compromised_nodes(obs_cpy: np.ndarray) -> np.ndarray:
    return np.where(obs_cpy.diagonal() == 2)[0]
    

def filter_candidate_neighbors(obs: np.ndarray, global_obs: np.ndarray) -> set:
    neighbor_set = set()
    obs_cpy = obs.copy()

    compromised_nodes = np.where(obs_cpy.diagonal() == 2)[0]
    np.fill_diagonal(obs_cpy, 0)
    neighbors = np.where(obs_cpy[compromised_nodes] == 1)[1]

    global_obs_cpy = global_obs.copy()
    all_compromised = np.where(global_obs_cpy.diagonal() == 2)[0]
    np.fill_diagonal(global_obs_cpy, 0)
    all_neighbors = np.where(global_obs_cpy[all_compromised] == 1)[1]

    for link in neighbors:
        # check if link in global obs
        if link in all_neighbors:
            neighbor_set.add(link)

    # return index of neighbors
    return neighbor_set


def attacker_is_isolated(global_obs: np.ndarray) -> bool:
    global_obs_cpy = global_obs.copy()
    all_compromised = np.where(global_obs_cpy.diagonal() == 2)[0]
    np.fill_diagonal(global_obs_cpy, 0)
    all_neighbors = np.where(global_obs_cpy[all_compromised] == 1)[1]

    if len(all_neighbors) == 0:
        return True
    else:
        return False


def set_neighbors(obs: np.ndarray, target_node: int, value: int) -> np.ndarray:
    state = obs[target_node][target_node]
    obs[target_node] = value

    obs[:, target_node] = obs[target_node]

    obs[target_node][target_node] = state
    
    return obs


def topology_builder(topo_name):
    topology_selector = {
        "clique": build_clique,
        "grid": build_grid,
        "ring": build_ring,
        "linear": build_linear,
        "tree": build_tree,
        "random": build_random
    }
    return topology_selector[topo_name]


def build_clique(num_nodes: int, start_position: int) -> np.ndarray:
    obs = np.ones((num_nodes, num_nodes), dtype=np.int8)
    np.fill_diagonal(obs, 0)
    
    obs[start_position][start_position] = 3

    return obs


def build_grid(num_nodes: int, start_position: int) -> np.ndarray:
    obs = np.zeros((num_nodes, num_nodes), dtype=np.int8)

    assert (num_nodes % 2) == 0
    for i, col in enumerate(obs):
        neighbors = set()

        neighbors.add((i + 1) % num_nodes)
        neighbors.add((i - 1) % num_nodes)
        neighbors.add(num_nodes - i - 1)

        for neighbor in neighbors:
            col[neighbor] = 1

    obs[start_position][start_position] = 3

    return obs


def build_ring(num_nodes: int, start_position: int) -> np.ndarray:
    obs = np.zeros((num_nodes, num_nodes), dtype=np.int8)

    for i, col in enumerate(obs):
        col[(i + 1) % num_nodes] = 1
        col[(i - 1) % num_nodes] = 1
        obs[i] = col

    obs[start_position][start_position] = 3

    return obs


def build_linear(num_nodes: int, start_position: int) -> np.ndarray:
    obs = np.zeros((num_nodes, num_nodes), dtype=np.int8)

    for i, col in enumerate(obs):
        col[(i + 1) % num_nodes] = 1
        col[(i - 1) % num_nodes] = 1
        obs[i] = col

    obs[0][num_nodes - 1] = 0
    obs[num_nodes - 1][0] = 0

    obs[start_position][start_position] = 3

    return obs


def build_tree(num_nodes: int, start_position: int) -> np.ndarray:
    obs = np.zeros((num_nodes, num_nodes), dtype=np.int8)

    assert (num_nodes % 4) == 0

    for i, col in enumerate(obs):
        col[(i + 1) % num_nodes] = 1
        col[(i - 1) % num_nodes] = 1
        obs[i] = col

    obs[0][num_nodes - 1] = 0
    obs[num_nodes - 1][0] = 0

    obs[start_position][start_position] = 3

    return obs


def build_random(num_nodes: int, start_position: int) -> np.ndarray:
    obs = np.zeros((num_nodes, num_nodes), dtype=np.int8)
    rng = default_rng()

    for i, col in enumerate(obs):
        if i + 1 < num_nodes:
            col[(i + 1) % num_nodes] = 1
        if i - 1 >= 0:
            col[(i - 1) % num_nodes] = 1

        for j in range(i + 2, len(obs)):
            link_probs = rng.binomial(1, p=min(NETWORK_SAMPLE_THRESHOLD, rng.random()))
            if link_probs == 1:
                col[j] = 1

        obs[i] = col
        obs[:, i] = col

    obs[start_position][start_position] = 3

    return obs


def mask_target_action(node_states: tf.Tensor, target_action_mask: tf.Tensor, state: int, action: int) -> tf.Tensor:
    indices = tf.where(tf.equal(node_states, state))[:, 0]
    action_batch_index = tf.unique(indices).y

    action_batch_index = tf.reshape(action_batch_index, (action_batch_index.shape[0], 1))

    action_selected = tf.reshape(tf.fill(action_batch_index.shape[0], action), (action_batch_index.shape[0], 1))
    action_selected = tf.cast(action_selected, dtype=tf.int64)

    action_indices = tf.keras.layers.concatenate([action_batch_index, action_selected], axis=1)

    target_action_mask = tf.tensor_scatter_nd_update(target_action_mask, action_indices, tf.ones(action_indices.get_shape()[0]))
    return target_action_mask


def set_target_node_mask(attacker_obs: tf.Tensor, mask: tf.Tensor):
    bool_mask = tf.math.reduce_any(tf.equal(attacker_obs, 2), axis=1)
    masked_rows = tf.boolean_mask(attacker_obs, bool_mask)
    action_possible = tf.where(tf.equal(attacker_obs, 1))

    action_batch_shape = (action_possible.get_shape()[0], 1)
    action_batch_index = tf.reshape(action_possible[:, 0],
                                    action_batch_shape)
    action_possible_index = tf.reshape(action_possible[:, 2],
                                    action_batch_shape)

    action_possible = tf.keras.layers.concatenate([action_batch_index, action_possible_index], axis=1)


    return tf.tensor_scatter_nd_update(mask, action_possible, tf.ones(action_possible.get_shape()[0]))


def elo_score(rating1: float, rating2: float, k: float, is_winner: bool):
    def win_probability(rating1: float, rating2: float):
        """Probability of agent with rating2 beating agent with rating1."""
        return 1.0 / (1 + math.pow(10, (rating2 - rating1) / 400))

    prob = win_probability(rating1, rating2)
 
    # Case -1 When Player A wins
    # Updating the Elo Ratings
    if is_winner:
        new_rating = min(max(0, rating1 + k * (1 - prob)), 2400)
    else:
        new_rating = min(max(0, rating1 + k * (0 - prob)), 2400)

    return new_rating
