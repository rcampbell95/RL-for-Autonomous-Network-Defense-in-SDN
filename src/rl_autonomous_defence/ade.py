from gym.spaces import Box, Discrete, Tuple
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
import os
import random

import pygame
from pygame import gfxdraw

import utils

class AutonomousDefenceEnv(AECEnv):
    '''
    The metadata holds environment constants. From gym, we inherit the "render.modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allosws the environment to be pretty printed.
    '''
    metadata = {'render.modes': ['rgb_array', 'human'], "name": "ad_v1", "emulate_network": False}

    def __init__(self):
        '''
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        '''
        self.action_out = os.getenv("RL_SDN_ACTIONSPACE", "multi").strip()
        self.multi_action = True if self.action_out == "autoreg" or self.action_out == "multi" else False
        self.possible_agents = ["attacker", "defender"]
        self.num_nodes = int(os.getenv("RL_SDN_NETWORKSIZE", "8").strip())
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.observation_spaces =  {agent: Box(-1, 2, (self.num_nodes, self.num_nodes)) for agent in self.possible_agents}
        self.NUM_ITERS = int(os.getenv("RL_SDN_HORIZON", "200").strip())
        self.defender_action_costs = {0: 1, 1: 6, 2:7}
        self.screen = None
        self.isopen = True

        if self.action_out == "multi" or self.action_out == "autoreg":
            self.action_spaces = {agent: Tuple([Discrete(self.num_nodes), Discrete(3)]) for agent in self.possible_agents}
        elif self.action_out == "single":
            self.action_spaces = {agent: Discrete(self.num_nodes * 3) for agent in self.possible_agents}


    def observation_space(self, agent):
        if "attacker" in agent:
            return self.observation_spaces["attacker"]
        elif "defender" in agent:
            return self.observation_spaces["defender"]
        else:
            return self.observation_spaces["attacker"]


    
    def action_space(self, agent):
        if "attacker" in agent:
            return self.action_spaces["attacker"]
        elif "defender" in agent:
            return self.action_spaces["defender"]
        else:
            return self.action_spaces["attacker"]


    def render(self, mode="rgb_array"):
        '''
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up sofme other display that a human can see and understand.
        '''
        screen_width = 600
        screen_height = 600

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        surf = pygame.Surface((screen_width, screen_height))
        surf.fill((255, 255, 255))

        node_colors = {
            -1: np.array([0, 255, 0], dtype=int),
            0: np.array([255, 255, 255], dtype=int),
            1: np.array([255, 255, 0], dtype=int),
            2: np.array([255, 0, 0], dtype=int),
        }

        angles = np.linspace(0, 2*np.math.pi, self.num_nodes + 1)
        angles = angles[:-1]
        angles = angles + (np.math.pi / self.num_nodes)
        r = 200 

        def_state = self.observations["defender"].diagonal()
        state = self.global_state["networkGraph"].diagonal()
        adj = self.global_state["networkGraph"]
        for i, theta in enumerate(angles):
            x1 = int(r * np.math.cos(theta)) + (screen_width // 2)
            y1 = int(r *  np.math.sin(theta)) +  (screen_height // 2)

            for j, theta2 in enumerate(angles):
                x2 = int(r * np.math.cos(theta2)) + (screen_width // 2)
                y2 = int(r *  np.math.sin(theta2)) +  (screen_height // 2)

                if adj[i][j] == 1:
                    pygame.draw.aalines(surf, points=[(x1, y1), (x2, y2)], closed=False, color=(0, 0, 0))

        for i, theta in enumerate(angles):
            x1 = int(r * np.math.cos(theta)) + (screen_width // 2)
            y1 = int(r *  np.math.sin(theta)) +  (screen_height // 2)

            gfxdraw.aacircle(
                surf, x1, y1, 21, (128, 128, 128)
            )
            if def_state[i] == -1 and state[i] == 1:
                gfxdraw.filled_circle(
                    surf, x1, y1, 20, (200, 255, 0)
                )
            elif def_state[i] == -1 and state[i] == 2:
                gfxdraw.filled_circle(
                    surf, x1, y1, 20, (255, 255, 255)
                )
            elif def_state[i] == -1 and state[i] == 0:
                gfxdraw.filled_circle(
                    surf, x1, y1, 20, node_colors[-1]
                )
            else:
                gfxdraw.filled_circle(
                    surf, x1, y1, 20, node_colors[state[i]]
                )


        surf = pygame.transform.flip(surf, False, True)
        self.screen.blit(surf, (0, 0))

        if mode == "human":
            pygame.display.flip()
            return self.isopen
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )


    def observe(self, agent):
        '''
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        '''
        # observation of one agent is the previous state of the other
        return np.array(self.observations[agent])


    def close(self):
        '''
        Close should release any graphical displays, subprocresses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        '''
        if self.screen is not None:
            pygame.quit()
            self.isopen = False

    def reset(self):
        '''
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - dones
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.

        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        '''
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # TODO -- reset state and observation accounting for initiall ycompromised nodes
        start_positions = random.sample([i for i in range(self.num_nodes)], 2)
        self.start_positions = {"attacker": start_positions[0], "defender": start_positions[1]}
        self.global_state = {}
        self.global_state["networkGraph"] = np.zeros((self.num_nodes, self.num_nodes))
        self.global_state["vulnMetrics"] = self._set_vuln_metrics(self.num_nodes)
        self.observations = self.reset_observations()
        self.episode_costs = []
        self.valid_actions = {agent: 0 for agent in self.agents}

        # Set initially compromised nodes
        self.num_moves = 0
        self.action_completed = False
        self.winner = "draw"
        '''
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        '''
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()


    def step(self, action):
        '''
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - dones
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        '''
        if self.dones[self.agent_selection]:
            # handles stepping an agent which is already done
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next done agent,  or if there are no more done agents, to the next live agent
            return self._was_done_step(action)

        agent = self.agent_selection


        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0

        # stores action of current agent

        self.action_completed = False
        # TODO -- update state in a reasonable way
        # State is global: i.e. it is thel global view of the environment
        self.observations[self.agent_selection] = self.update_observation(self.observations[agent], agent, action)
    
        self.num_moves += 1
        
        if self._check_win_conditions(agent):
            self.dones = {agent: True for agent in self.possible_agents}
        #else:
            #self.dones = {agent: self.num_moves >= self.NUM_ITERS for agent in self.possible_agents}

        if self.metadata["emulate_network"]:
            self.send_message_to_controller(action)
        # collect reward if it is the last agent to act
        ## if self._agent_selector.is_last():
            # rewards for all agents are placed in the .rewards dictionary
            # TODO -- Add reward function
            # REWARD_MAP[(self.state[self.agents[0]], self.state[self.agents[1]])]
            # The dones dictionary must be updated for all players.

        #for agent_i, done in zip(self.agents, self.dones):
        self.rewards = self.reward()

            # observe the current state
            #for i in self.agents:
            #    self.observations[i] = self.state[self.agents[self.agent_name_mapping[i]]]
        #else:
            # necessary so that observe() returns a reasonable observation at all times.
            ## self.state[self.agents[1 - self.agent_name_mapping[agent]]] = NONE
            # no rewards are allocated until both players give an action
        #    pass
            #self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

    def _set_vuln_metrics(self, num_nodes):
        metrics = {"impactScore": [], "exploitabilityScore": [], "baseScore": []}

        self.mean_impact_score = float(os.getenv("RL_SDN_MIS", 4.311829))
        self.mean_exploitability_score = float(os.getenv("RL_SDN_MES", 2.592744))
        self.std_impact_score = float(os.getenv("RL_SDN_STDIS", 1.539709))
        self.std_exploitability_score = float(os.getenv("RL_SDN_STDES", 0.954755))

        for _ in range(num_nodes):
            # Baseline impact: mean = 4.311829, std = 1.539709
            # Baseline exploitability: mean = 2.592744, std = 0.954755

            impact_score = np.random.normal(self.mean_impact_score, self.std_impact_score)
            exploitability_score = np.random.normal(self.mean_exploitability_score, self.std_exploitability_score)

            base_score = min(round(1.08 * impact_score + exploitability_score), 10)

            metrics["impactScore"].append(impact_score)
            metrics["exploitabilityScore"].append(exploitability_score)
            metrics["baseScore"].append(base_score)

        return metrics


    def _update_global_obs(self, state):
        node_states = self.global_state["networkGraph"].diagonal()
        self.global_state["networkGraph"] = np.logical_or(self.global_state["networkGraph"], state).astype("int32")
        self.global_state["networkGraph"][np.diag_indices(self.num_nodes)] = node_states
        self.global_state["networkGraph"][np.diag_indices(self.num_nodes)] += state[np.diag_indices(self.num_nodes)].astype("int32")


    def _validate_attacker_action(self, action, state):
        explore_topo = (action == 0) and (state == 2) and utils.string_to_bool(os.getenv("RL_SDN_EXPLORETOPO", "True").strip())
        scan_vulns = (action == 1) and (state == 0) and utils.string_to_bool(os.getenv("RL_SDN_SCANVULN", "True").strip())
        attack_vulns = (action == 2) and (state == 1) and utils.string_to_bool(os.getenv("RL_SDN_ATTACKVULN", "True").strip())
        
        return scan_vulns or attack_vulns or explore_topo


    def _validate_defender_action(self, action, state):
        check_status = (action == 0) and utils.string_to_bool(os.getenv("RL_SDN_CHECKSTATUS", "True").strip())
        isolate_node = (action == 1) and (state == 2) and utils.string_to_bool(os.getenv("RL_SDN_ISOLATENODE", "True").strip())
        move_flag = (action == 2) and (state == 0) and utils.string_to_bool(os.getenv("RL_SDN_MOVEFLAG", "True").strip())

        return check_status or isolate_node or move_flag


    def _check_win_conditions(self, agent):
        if agent == "attacker":
            flag_node_index = np.where(self.observations["defender"] == -1)[0]
            flag_node_compromised = self.global_state["networkGraph"].diagonal()[flag_node_index] == 2
            if flag_node_compromised:
                self.winner = agent
                return True
        elif agent == "defender":
            candidate_neighbors = utils.filter_candidate_neighbors(self.observations["attacker"], self.global_state["networkGraph"])
            attacker_no_valid_moves = len(candidate_neighbors) == 0
            if attacker_no_valid_moves:
                self.winner = agent
                return True

        if self.num_moves >= self.NUM_ITERS:
            self.winner = "draw"
            return True
            #candidate_neighbors = self._filter_candidate_neighbors(self.observations["attacker"])

            #if len(candidate_neighbors) == 0:
            #    self.winner = agent
            #    return True
        
        return False

    def _process_action(self, action):
        if self.multi_action:
            target_node = action[0]
            target_action = action[1]
        else:
            target_node = action % self.num_nodes 
            target_action = action // self.num_nodes

        return target_node, target_action


    def _defender_step(self, obs, action):
        target_node, target_action = self._process_action(action)

        base_score = self.global_state["vulnMetrics"]["baseScore"][target_node]

        if target_action == 0:
            success = np.random.random()

            true_state = self.global_state["networkGraph"][target_node][target_node]
            if true_state == 2:
                threshold = base_score / (10 * np.log2(self.num_nodes))

                if success < threshold:
                   obs[target_node][target_node] = 2
                   self.episode_costs.append(self.defender_action_costs[target_action])


        elif target_action == 1:

            obs = utils.set_neighbors(obs, target_node, np.array([0] * self.num_nodes))

            self.global_state["networkGraph"][target_node] = obs[target_node]
            self.global_state["networkGraph"][:, target_node] = obs[target_node]
            self.episode_costs.append(self.defender_action_costs[target_action])
        elif target_action == 2:
            success = np.random.random()

            # check state
            true_state = self.global_state["networkGraph"][target_node][target_node]
            if true_state == 2:
                threshold = base_score / (10 * np.log2(self.num_nodes))
                if success < threshold:
                    obs[target_node][target_node] = 2

            else:
                # migrate critical node
                critical_node = np.where(obs == -1)[0]
                critical_node = critical_node.item()

                assert critical_node < self.num_nodes
                try:
                    obs[critical_node][critical_node] = 0
                    #self.global_state["networkGraph"][critical_node][critical_node] = 0
                except:
                    raise AssertionError(f"{critical_node}, {obs.shape}")

                obs[target_node][target_node] = -1

                self.episode_costs.append(self.defender_action_costs[target_action])


        return obs


    def _attacker_step(self, obs, action):
        target_node, target_action = self._process_action(action)

        # get exploitability score
        exploitability = self.global_state["vulnMetrics"]["exploitabilityScore"][target_node] / 10

        if target_action == 0:
            obs[target_node] = self.global_state["networkGraph"][target_node]
            obs[:, target_node] = self.global_state["networkGraph"][target_node]
        else:
            # Actions 1 and 2; Identify and target vulns, respectively
            # generate random number
            action_score = np.random.random()
            # if calculated prob < exploitability score of node
            if action_score < exploitability:
                # apply action to target node
                obs[target_node][target_node] = target_action
                # update global state
                self.global_state["networkGraph"][target_node][target_node] = target_action
        return obs


    def reset_observations(self):
        topology = os.getenv("RL_SDN_TOPOLOGY", "clique").strip()
        topo_builder = utils.topology_builder(topology)
        defender_obs = topo_builder(self.num_nodes, self.start_positions["defender"])
        # Reset state per agent

        compromised_node_position = self.start_positions["attacker"]
        attacker_obs = np.zeros((self.num_nodes, self.num_nodes))

        utils.set_neighbors(attacker_obs, compromised_node_position, defender_obs[compromised_node_position])

        attacker_obs[compromised_node_position][compromised_node_position] = 2

        self.global_state["networkGraph"] = defender_obs.copy()
        np.fill_diagonal(self.global_state["networkGraph"], 0)
        self.global_state["networkGraph"][compromised_node_position][compromised_node_position] = 2

        return {"defender": defender_obs, "attacker": attacker_obs}


    def reward_selector(self, reward_specifier):
        reward_funcs = {
            "gabirondo": self.reward_gabirondo,
            "reward-costs-penalty": self.reward_costs_penalty,
            "reward-costs-clipped": self.reward_costs_clipped,
            "reward-impact-penalty": self.reward_impact_penalty,
            "reward-binary": self.reward_binary
        }

        return reward_funcs[reward_specifier]


    def reward_gabirondo(self, total_impact, total_cost):
        moves_made = self.NUM_ITERS - self.num_moves

        attacker_reward = 5 * total_impact
        defender_reward = max(0, moves_made - 5 * total_impact - total_cost)
        return {"attacker": attacker_reward, "defender": defender_reward}



    def reward_costs_penalty(self, total_impact, total_cost):
        defender_reward = self.mean_impact_score * self.num_nodes - total_impact - 0.1 * total_cost

        return {"attacker": total_impact, "defender": defender_reward}



    def reward_costs_clipped(self, total_impact, total_cost):
        defender_reward = max(0, self.mean_impact_score * self.num_nodes - total_impact - 0.1 * total_cost)

        return {"attacker": total_impact, "defender": defender_reward}


    
    def reward_impact_penalty(self, total_impact, total_cost):
        defender_reward = self.mean_impact_score * self.num_nodes - total_impact - 0 * total_cost
        return {"attacker": total_impact, "defender": defender_reward}


    def reward_binary(self, total_impact, total_cost):
        reward = {"attacker": 0, "defender": 0}

        if all(self.dones.values()) and self.winner != "draw":
            reward[self.winner] = 1
            reward[self.agents[1 - self.agent_name_mapping[self.winner]]] = 0

        return reward


    def reward(self):
        reward = {"attacker": self.valid_actions["attacker"], "defender": self.valid_actions["defender"]}

        self.valid_actions = {agent: 0 for agent in self.agents}

        exploited_nodes = np.where(self.global_state["networkGraph"] == 2)[0]
        impact_scores = [self.global_state["vulnMetrics"]["impactScore"][node] for node in exploited_nodes] 
        total_impact = sum(impact_scores)
        total_cost = sum(self.episode_costs)
        defender_reward_func = self.reward_selector(os.getenv("RL_SDN_REWARD", "reward-costs-clipped").strip())

        rewards = defender_reward_func(total_impact, total_cost)

        winner_rewards = {"attacker": rewards["attacker"], "defender": rewards["defender"]}

        if all(self.dones.values()) and self.winner != "draw":
            reward[self.winner] = winner_rewards[self.winner]
            reward[self.agents[1 - self.agent_name_mapping[self.winner]]] = -1 * reward[self.winner]

        return reward

    def update_observation(self, obs, agent, action):
        target_node, target_action = self._process_action(action)

        if agent == "attacker":
            # choose a random node from list of neighbors
            candidate_neighbors = utils.filter_candidate_neighbors(obs, self.global_state["networkGraph"])
            #neighbors = obs.diagonal()
            #is_compromised = neighbors == 2
            #indices = np.arange(neighbors.size)
            valid_action = self._validate_attacker_action(target_action, obs[target_node][target_node])
            valid_target = (target_node in candidate_neighbors) or obs[target_node][target_node] == 2
            action_possible = valid_target and valid_action

            # Explore the topology
            #for neighbor in neighbor_set:
            #    # if edge exists betwen target node and neighbor
            #    if self.global_state["networkGraph"][target_node][neighbor] == 1:
            #        action_possible = True
            #        break
            #
            #    else:
            #        # Is this check done for all nodes or only compromised nodes?
            #        # remove edge from adjacency matrix
            #        obs[target_node][neighbor] = 0

            if action_possible:
                obs = self._attacker_step(obs, action)
            else:
                self.valid_actions["attacker"] += -1
            #state = np.zeros(shape=(self.num_nodes, self.num_nodes), dtype=np.int32)
        elif agent == "defender":
            #state = np.zeros(shape=(self.num_nodes, self.num_nodes), dtype=np.int32)

            action_possible = self._validate_defender_action(target_action, obs[target_node][target_node])
            
            if action_possible:
                obs = self._defender_step(obs, action)
            else:
                self.valid_actions["defender"] += -1

        return obs


    def send_message_to_controller(self, action):
        # Map action to SDN action for controller
        # Check validity of action
        # Report status of action taken   
        pass


def env():
    '''
    The env function wraps the environment in 3 wrappers by default. These
    wrappers contain logic that is common to many pettingzoo environments.
    We recommend you use at least the OrderEnforcingWrapper on your own environment
    to provide sane error messages. You can find full documentation for these methods
    elsewhere in the developer documentation.
    '''
    env = AutonomousDefenceEnv()
    env = wrappers.CaptureStdoutWrapper(env)
    #env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env