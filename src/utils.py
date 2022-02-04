import numpy as np


def select_policy(agent_id):
    return agent_id


def string_to_bool(bool_string):
    if isinstance(bool_string, bool):
        return bool_string

    if bool_string.lower() == "false":
        return False
    elif bool_string.lower() == "true":
        return True


def filter_candidate_neighbors(obs, global_obs):
    neighbor_set = set()
    # explored_or_compromised = obs[np.where(obs == filter_state)[0]]
    # A node may be visited but not have any neighbors?
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

        # if in global obs, add to neighbor set
        # else update observation in global obs

    # return index of neighbors
    return neighbor_set


def topology_builder(topo_name):
    topology_selector = {
        "clique": build_clique,
        "grid": build_grid,
        "linear": build_linear
    }
    return topology_selector[topo_name]


def build_clique(num_nodes, start_position):
    obs = np.ones((num_nodes, num_nodes))
    np.fill_diagonal(obs, 0)
    
    obs[start_position][start_position] = -1

    return obs


def build_grid(num_nodes, start_position):
    obs = np.zeros((num_nodes, num_nodes))

    neighbors = set()

    assert (num_nodes % 2) == 0
    
    for i, col in enumerate(obs):
        neighbors.add((i + 1) % num_nodes)
        neighbors.add((i - 1) % num_nodes)
        neighbors.add(num_nodes - i - 1)

        for neighbor in neighbors:
            col[neighbor] = 1
    
    obs[start_position][start_position] = -1

    return obs


def build_linear(num_nodes, start_position):
    obs = np.zeros((num_nodes, num_nodes))
    
    for i, col in enumerate(obs):
        col[(i + 1) % num_nodes] = 1
        col[(i - 1) % num_nodes] = 1
        obs[i] = col
    
    obs[start_position][start_position] = -1

    return obs

