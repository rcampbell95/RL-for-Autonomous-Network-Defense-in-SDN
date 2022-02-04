from gym.spaces import Box, Discrete, Tuple
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
import os
import random

import utils

class AutonomousDefenceEnv(AECEnv):
    '''
    The metadata holds environment constants. From gym, we inherit the "render.modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allosws the environment to be pretty printed.
    '''
    metadata = {'render.modes': ['human'], "name": "ad_v1", "emulate_network": False}

    def __init__(self):
        '''
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        '''
        self.possible_agents = ["attacker", "defender"]
        self.num_nodes = int(os.getenv("RL_SDN_NETWORKSIZE", "8").strip())
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.observation_spaces =  {agent: Box(-1, 2, (self.num_nodes, self.num_nodes)) for agent in self.possible_agents}
        self.action_spaces = {agent: Discrete(self.num_nodes * 3) for agent in self.possible_agents}
        self.NUM_ITERS = int(os.getenv("RL_SDN_HORIZON", "200").strip())
        self.defender_action_costs = {0: 1, 1: 6, 2:7, 3:5}


    def observation_space(self, agent):
        return self.observation_spaces[agent]

    
    def action_space(self, agent):
        return self.action_spaces[agent]


    def render(self, mode="human"):
        '''
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up sofme other display that a human can see and understand.
        '''
        if not all(self.dones.values()):
            attacker_state = self.observations[self.agents[0]].diagonal()
            defender_state = self.observations[self.agents[1]].diagonal()
            global_state = self.global_state["networkGraph"].diagonal()
            game_state = (f"Current state: Attacker: {attacker_state} , Defender: {defender_state}, Global state: {global_state}")
        elif all(self.dones.values()):
            game_state = f"Game over: winner is {self.winner}"
        print(f"Currently selected agent: {self.agent_selection}")
        print(game_state)

 
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
        pass


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
        self.observations = {agent: self.reset_observation(agent) for agent in self.agents}


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
        # State is global: i.e. it is the global view of the environment
        self.observations[self.agent_selection] = self.update_observation(self.observations[agent], agent, action)
    
        self.num_moves += 1
        
        if self._check_win_conditions(agent):
            self.dones = {agent: True for agent in self.possible_agents}
        #else:
            #self.dones = {agent: self.num_moves >= self.NUM_ITERS for agent in self.possible_agents}

        #self.render()

        if self.metadata["emulate_network"]:
            self.send_message_to_controller(action)
        # collect reward if it is the last agent to act
        ## if self._agent_selector.is_last():
            # rewards for all agents are placed in the .rewards dictionary
            # TODO -- Add reward function
            # REWARD_MAP[(self.state[self.agents[0]], self.state[self.agents[1]])]
            # The dones dictionary must be updated for all players.

        #for agent_i, done in zip(self.agents, self.dones):
        self.rewards = self.reward_nice()

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


    def _set_neighbors(self, obs, target_node, value):
        obs[target_node] = value

        obs[:, target_node] = obs[target_node]
        
        return obs


    def _check_win_conditions(self, agent):
        if agent == "attacker":
            flag_node_index = np.where(self.observations["defender"] == -1)[0]
            flag_node_compromised = self.global_state["networkGraph"].diagonal()[flag_node_index] == 2
            if flag_node_compromised:
                self.winner = agent
                return True
        elif agent == "defender":
            candidate_neighbors = utils.filter_candidate_neighbors(self.global_state["networkGraph"], self.global_state["networkGraph"])
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

    def _defender_step(self, obs, action):
        target_node = action % self.num_nodes 
        target_action = action // self.num_nodes
        base_score = self.global_state["vulnMetrics"]["baseScore"][target_node]

        if target_action == 0:
            success = np.random.random()

            true_state = self.global_state["networkGraph"][target_node][target_node]
            if true_state == 2:
                threshold = base_score / (10 * np.log2(self.num_nodes))
                obs[target_node][target_node] = 2 if success < threshold else obs[target_node][target_node]

        elif target_action == 1:

            obs = self._set_neighbors(obs, target_node, np.array([0] * self.num_nodes))

            self.global_state["networkGraph"][target_node] = obs[target_node]
            self.global_state["networkGraph"][:, target_node] = obs[target_node]
        elif target_action == 2:
            success = np.random.random()

            # check state
            true_state = self.global_state["networkGraph"][target_node][target_node]
            if true_state == 2:
                threshold = base_score / (10 * np.log2(self.num_nodes))
                obs[target_node][target_node] = 2 if success < threshold else obs[target_node][target_node]

            else:
                # migrate critical node
                critical_node = np.where(obs == -1)[0]
                critical_node = critical_node.item()

                assert critical_node < self.num_nodes
                try:
                    obs[critical_node][critical_node] = 0
                except:
                    raise AssertionError(f"{critical_node}, {obs.shape}")



                obs[target_node][target_node] = -1

        return obs


    def _attacker_step(self, obs, action):
        target_node = action % self.num_nodes
        target_action = action // self.num_nodes
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


    def reset_observation(self, agent):
        # Reset state per agent
        if agent == "attacker":
            obs = np.zeros((self.num_nodes, self.num_nodes))
            compromised_node_position = self.start_positions[agent]
            obs[compromised_node_position][compromised_node_position] = 2 

            obs = self._set_neighbors(obs, compromised_node_position, np.array([1] * self.num_nodes))
 
        elif agent == "defender":
            #obs = np.ones((self.num_nodes, self.num_nodes))
            #np.fill_diagonal(obs, 0)
            
            topo_builder = utils.topology_builder("clique")
            obs = topo_builder(self.num_nodes, self.start_positions[agent])

        self._update_global_obs(obs)

        return obs


    def reward_nice(self):
        reward = {"attacker": 0, "defender": 0}

        exploited_nodes = np.where(self.global_state["networkGraph"] == 2)[0]
        impact_scores = [self.global_state["vulnMetrics"]["impactScore"][node] for node in exploited_nodes] 
        total_impact = sum(impact_scores) * 10

        attack_reward = total_impact
        def_reward = 10 * self.mean_impact_score * self.num_nodes - total_impact

        winner_rewards = {"attacker": attack_reward, "defender": def_reward}

        if all(self.dones.values()) and self.winner != "draw":
            reward[self.winner] = winner_rewards[self.winner]
            reward[self.agents[1 - self.agent_name_mapping[self.winner]]] = -1 * reward[self.winner]

        return reward


    def reward_agents(self, agent, done):
        agent_won_episode = done and self.winner == agent
        reward = {}

        if agent_won_episode:
            reward[agent] = 1
            reward[self.agents[1 - self.agent_name_mapping[agent]]] = 0
        elif not agent_won_episode:
            reward[agent] = 0
            reward[self.agents[1 - self.agent_name_mapping[agent]]] = 0
        return reward

    def update_observation(self, obs, agent, action):
        target_node = action % self.num_nodes
        target_action = action // self.num_nodes
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
            #state = np.zeros(shape=(self.num_nodes, self.num_nodes), dtype=np.int32)
        elif agent == "defender":
            #state = np.zeros(shape=(self.num_nodes, self.num_nodes), dtype=np.int32)

            action_possible = self._validate_defender_action(target_action, obs[target_node][target_node])
            
            if action_possible:
                obs = self._defender_step(obs, action)
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