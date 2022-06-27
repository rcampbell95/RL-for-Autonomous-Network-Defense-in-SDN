import numpy as np
from numpy.random import default_rng
import os


NETWORK_SAMPLE_THRESHOLD = float(os.getenv("RL_SDN_STDIS", "0.01"))


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
