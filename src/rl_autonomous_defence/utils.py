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




def set_target_node_mask(attacker_obs: tf.Tensor, features: tf.Tensor=None):
    if features is not None:
        assert attacker_obs.shape[0] == features.shape[0]
        assert attacker_obs.shape[-1] == features.shape[1]
        obs = tf.linalg.set_diag(attacker_obs, features)
    else:
        obs = attacker_obs
    bool_mask = tf.cast(tf.math.reduce_any(tf.equal(obs, 2), axis=1), tf.float32)
    rows = tf.math.multiply(obs, tf.expand_dims(bool_mask, -1))

    summed_rows = tf.reduce_sum(rows, axis=1)

    assert summed_rows.shape[1] == attacker_obs.shape[-1]

    if 0 in summed_rows.shape:
        return tf.zeros([attacker_obs.shape[0], attacker_obs.shape[-1]])
    elif summed_rows.shape[0] == attacker_obs.shape[0]:
        return tf.cast(tf.cast(tf.not_equal(summed_rows, 0), tf.int32), tf.float32)
    try:
        assert summed_rows.shape[0] == attacker_obs.shape[0]
        print(f"Mismatch in obs shape and masked rows. Obs shape {attacker_obs.shape} Masked rows shape: {masked_rows.shape}")
    except AssertionError:
        raise Exception(f"Mismatch in obs shape and masked rows. Obs shape {attacker_obs.shape} Masked rows shape: {masked_rows.shape} Observation: {attacker_obs}")

    #action_possible = tf.where(tf.equal(attacker_obs, 1))

    #action_batch_shape = (action_possible.get_shape()[0], 1)
    #action_batch_index = tf.reshape(action_possible[:, 0],
    #                                action_batch_shape)
    #action_possible_index = tf.reshape(action_possible[:, 2],
    #                                action_batch_shape)

    #action_possible = tf.keras.layers.concatenate([action_batch_index, action_possible_index], axis=1)


    #return tf.tensor_scatter_nd_update(mask, action_possible, tf.ones(action_possible.get_shape()[0]))


def mask_attacker_actions(observation: tf.Tensor, features: tf.Tensor=None) -> tf.Tensor:
    agent_target_action_mask = tf.zeros((observation.shape[0], 3))
    if features is not None:
        diagonals = features
    else:
        diagonals = tf.linalg.diag_part(observation)

    agent_target_node_mask = set_target_node_mask(observation, features)

    # explore topology
    agent_target_action_mask = mask_target_action(diagonals, agent_target_action_mask, 2, 0)

    # scan vuln
    agent_target_action_mask = mask_target_action(diagonals, agent_target_action_mask, 0, 1)
    # compromise vuln
    agent_target_action_mask = mask_target_action(diagonals, agent_target_action_mask, 1, 2)

    if train_config["agent"]["action_space"] == "multi":
        action_mask = tf.keras.layers.concatenate([agent_target_node_mask, agent_target_action_mask])
    elif train_config["agent"]["action_space"] == "product":
        num_actions = 3

        assert agent_target_node_mask.shape[1] == observation.shape[-1]
        tiled_action_mask = tf.tile(agent_target_node_mask, [1, num_actions])
        assert tiled_action_mask.shape[0] == observation.shape[0]
        assert tiled_action_mask.shape[-1] == observation.shape[-1] * num_actions
        action_mask = tf.reshape(tiled_action_mask, [observation.shape[0], num_actions, observation.shape[-1]])
        action_mask = tf.math.multiply(action_mask, tf.expand_dims(agent_target_action_mask, axis=-1))
        action_mask = tf.reshape(action_mask,  [observation.shape[0], -1])
    return action_mask

def mask_defender_actions(observation: tf.Tensor, features: tf.Tensor=None) -> tf.Tensor:
    agent_target_action_mask = tf.zeros((observation.shape[0], 3))
    if features is not None:
        diagonals = features
    else:
        diagonals = tf.linalg.diag_part(observation)
    agent_target_node_mask = tf.ones((observation.shape[0], observation.shape[-1]))

    is_critical_indices = tf.where(tf.equal(diagonals, 3))

    agent_target_node_mask = tf.tensor_scatter_nd_update(agent_target_node_mask, is_critical_indices, tf.zeros(is_critical_indices.get_shape()[0]))

    current_size = train_config["environment"]["network_size"] #int(float(os.environ["RL_SDN_NETWORKSIZE"]))
    max_size = train_config["environment"]["network_size_max"] #int(os.environ.get("RL_SDN_NETWORKSIZE-MAX", str(current_size)))
    assert isinstance(current_size, int)
    assert isinstance(max_size, int)

    if max_size > current_size:
        indices = [[[j, i] for i in range(current_size, max_size)] for j in range(agent_target_node_mask.shape[0])]
        updates = [[0 for i in range(len(indices[0]))] for j in range(agent_target_node_mask.shape[0])]

        indices = tf.constant(indices, dtype=tf.int32)
        updates = tf.constant(updates, dtype=tf.float32)

        agent_target_node_mask = tf.tensor_scatter_nd_update(agent_target_node_mask, indices, updates)

    # migrate node
    agent_target_action_mask = mask_target_action(diagonals, agent_target_action_mask, 0, 2)
    # check status
    agent_target_action_mask = mask_target_action(diagonals, agent_target_action_mask, 0, 0)
    # isolate node
    agent_target_action_mask = mask_target_action(diagonals, agent_target_action_mask, 2, 1)

    if train_config["agent"]["action_space"] == "multi":
        action_mask = tf.keras.layers.concatenate([agent_target_node_mask, agent_target_action_mask])
    elif train_config["agent"]["action_space"] == "product":
        num_actions = 3
        action_mask = tf.reshape(tf.tile(agent_target_node_mask, [1, num_actions]), [observation.shape[0], num_actions, observation.shape[-1]])
        action_mask = tf.math.multiply(action_mask, tf.expand_dims(agent_target_action_mask, axis=-1))
        action_mask = tf.reshape(action_mask,  [observation.shape[0], -1])
    return action_mask


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


def normalize_batch(obs_batch: tf.Tensor) -> tf.Tensor:
    adj_batch_tensor = obs_batch
    epsilon = 1e-6

    node_degrees = tf.reduce_sum(adj_batch_tensor, axis=-2) + epsilon
    pow_tensor = tf.ones(node_degrees.shape) * -0.5

    pow_tensor = tf.math.pow(node_degrees, pow_tensor)
    zeros_tensor = tf.zeros(adj_batch_tensor.shape, dtype=tf.float32)
    normal_diag = tf.linalg.set_diag(zeros_tensor, pow_tensor)

    num_dims = [i for i in range(len(adj_batch_tensor.shape))]
    permute_dims = num_dims[:-2] + [num_dims[-1], num_dims[-2]]

    normalized_adjacency = tf.transpose(tf.matmul(adj_batch_tensor, normal_diag), perm=permute_dims)
    return tf.matmul(normalized_adjacency, normal_diag)