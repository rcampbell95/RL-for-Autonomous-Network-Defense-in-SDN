from gym.spaces import Box, Discrete, Tuple
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers


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
        self.num_nodes = 5
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.observation_spaces =  {agent: Box(-1, 2, (self.num_nodes, self.num_nodes)) for agent in self.possible_agents}
        self.action_spaces = {agent: Tuple([Discrete(self.num_nodes), Discrete(3)]) for agent in self.possible_agents}
        self.NUM_ITERS = 100


    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        return self.action_spaces[agent]

    def render(self, mode="human"):
        '''
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        '''
        if len(self.agents) == 2:
            attacker_state = self.observations[self.agents[0]].diagonal()
            defender_state = self.observations[self.agents[1]].diagonal()
            global_state = self.global_state.diagonal()
            game_state = (f"Current state: Attacker: {attacker_state} , Defender: {defender_state}, Global state: {global_state}")
        else:
            game_state = "Game over"
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
        Close should release any graphical displays, subprocesses, network connections
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
        self.global_state = np.zeros((self.num_nodes, self.num_nodes))
        self.observations = {agent: self.reset_observation(agent) for agent in self.agents}
        # Set initially compromised nodes
        self.num_moves = 0
        self.action_completed = False
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

        self.render()


        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0

        # stores action of current agent

        self.action_completed = False
        # TODO -- update state in a reasonable way
        # State is global: i.e. it is the global view of the environment
        self.observations[self.agent_selection] = self.update_observation(agent, self.observations[agent], action)
    

        if self.metadata["emulate_network"]:
            self.send_message_to_controller(action)
        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            # rewards for all agents are placed in the .rewards dictionary
            # TODO -- Add reward function
            self.rewards[self.agents[0]], self.rewards[self.agents[1]] = (self.attacker_reward(self.observations["attacker"]), self.defender_reward(self.observations["defender"]))
            #REWARD_MAP[(self.state[self.agents[0]], self.state[self.agents[1]])]

            self.num_moves += 1
            # The dones dictionary must be updated for all players.
            self.dones = {agent: self.num_moves >= self.NUM_ITERS for agent in self.agents}

            # observe the current state
            #for i in self.agents:
            #    self.observations[i] = self.state[self.agents[self.agent_name_mapping[i]]]
        else:
            # necessary so that observe() returns a reasonable observation at all times.
            ## self.state[self.agents[1 - self.agent_name_mapping[agent]]] = NONE
            # no rewards are allocated until both players give an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()


    def _update_global_obs(self, state):
        node_states = self.global_state.diagonal()
        self.global_state = np.logical_or(self.global_state, state).astype("int32")
        self.global_state[np.diag_indices(self.num_nodes)] = node_states
        self.global_state[np.diag_indices(self.num_nodes)] += state[np.diag_indices(self.num_nodes)].astype("int32")


    def _validate_attacker_action(self, action, state):
        #explore_topo = (action == 0)
        scan_vulns = (action == 1) and (state == 0)
        attack_vulns = (action == 2) and (state == 1)
        
        return scan_vulns or attack_vulns


    def _validate_defender_action(self, action, state):
        check_status = (action == 0)
        isolate_node = (action == 1) and (state == 2)
        move_flag = (action == 2) and (state == 0)

        return check_status or isolate_node or move_flag


    def _filter_candidate_neighbors(self, obs):
        neighbor_set = set()
        #explored_or_compromised = obs[np.where(obs == filter_state)[0]]
        # A node may be visited but not have any neighbors?
        obs_cpy = obs.copy()
        compromised_nodes = np.where(obs_cpy.diagonal() == 2)[0]
        np.fill_diagonal(obs_cpy, 0)
        neighbors = np.where(obs[compromised_nodes] == 1)[1]

        for link in neighbors:
            neighbor_set.add(link)

        # return index of neighbors
        return neighbor_set

    def _defender_step(self, obs, action):
        target_node = action[0]
        target_action = action[1]
        # hardcode base score for now
        base_score = 0.5

        if target_action == 0:
            success = np.random.random()

            true_state = self.global_state[target_node][target_node]
            if true_state == 2:
                obs[target_node][target_node] = 2 if success < base_score else obs[target_node][target_node]

        elif target_action == 1:
            obs[target_node][:target_node] = 0
            obs[target_node][target_node + 1:] = 0

            obs[:, target_node] = obs[target_node]

            self.global_state[target_node] = obs[target_node]
            self.global_state[:, target_node] = obs[target_node]
        elif target_action == 2:
            pass         

        return obs

    def reset_observation(self, agent):
        # Reset state per agent
        if agent == "attacker":
            obs = np.zeros((self.num_nodes, self.num_nodes))
            obs[0][0] = 2 
            obs[0, 1:] = 1 
            obs[1:, 0] = 1

        elif agent == "defender":
            obs = np.ones((self.num_nodes, self.num_nodes))
            np.fill_diagonal(obs, 0)
            
            obs[self.num_nodes - 1][self.num_nodes - 1] = -1

        self._update_global_obs(obs)

        return obs

    def attacker_reward(self, obs):
        return obs.diagonal().mean()

    def defender_reward(self, obs):
        return -1 * self.attacker_reward(obs)

    def update_observation(self, agent, obs, action):
        target_node = action[0]
        target_action = action[1]

        if agent == "attacker":
            # choose a random node from list of neighbors
            candidate_neighbors = self._filter_candidate_neighbors(obs)
            #neighbors = obs.diagonal()
            #is_compromised = neighbors == 2
            #indices = np.arange(neighbors.size)
            valid_action = self._validate_attacker_action(target_action, obs[target_node][target_node])
            action_possible = (target_node in candidate_neighbors) and valid_action

            # Explore the topology
            #for neighbor in neighbor_set:
            #    # if edge exists betwen target node and neighbor
            #    if self.global_state[target_node][neighbor] == 1:
            #        action_possible = True
            #        break
            #
            #    else:
            #        # Is this check done for all nodes or only compromised nodes?
            #        # remove edge from adjacency matrix
            #        obs[target_node][neighbor] = 0

            if action_possible:
                # get exploitability score
                exploitability = 0.50
                # generate random number
                action_score = np.random.random()
                # if exploitability/ >= random number
                if action_score > exploitability:
                    # apply action to target node
                    self.action_completed = True
                    obs[target_node][target_node] = target_action
                    # update global state
                    self.global_state[target_node][target_node] = target_action

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