#!/usr/bin/env python
# coding: utf-8


from gym.spaces import Box, Discrete
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers

import ray.rllib.agents.pg as PG

from ray.tune.registry import register_env
# import the pettingzoo environment
# import rllib pettingzoo interface
from ray.rllib.env import PettingZooEnv

import logging




ROCK = 0
PAPER = 1
SCISSORS = 2
NONE = 3
MOVES = ["ROCK", "PAPER", "SCISSORS", "None"]
NUM_ITERS = 100
REWARD_MAP = {
    (ROCK, ROCK): (0, 0),
    (ROCK, PAPER): (-1, 1),
    (ROCK, SCISSORS): (1, -1),
    (PAPER, ROCK): (1, -1),
    (PAPER, PAPER): (0, 0),
    (PAPER, SCISSORS): (-1, 1),
    (SCISSORS, ROCK): (-1, 1),
    (SCISSORS, PAPER): (1, -1),
    (SCISSORS, SCISSORS): (0, 0),
}




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
        self.action_spaces = {agent: Discrete(4) for agent in self.possible_agents}

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
            string = ("Current state: Agent1: {} , Agent2: {}".format(MOVES[self.state[self.agents[0]]], MOVES[self.state[self.agents[1]]]))
        else:
            string = "Game over"
        print(string)

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
        self.state = {agent: self.reset_state(agent) for agent in self.agents}
        self.observations = {agent: self.reset_observation(agent) for agent in self.agents}
        # Set initially compromised nodes
        self.num_moves = 0
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

        # TODO -- update state in a reasonable way
        # State is global: i.e. it is the global view of the environment
        self.state[self.agent_selection] = self.state_update(agent, self.state[agent], action)
        
        if self.metadata["emulate_network"]:
            self.send_message_to_controller(action)
        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            # rewards for all agents are placed in the .rewards dictionary
            # TODO -- Add reward function
            self.rewards[self.agents[0]], self.rewards[self.agents[1]] = (self.attacker_reward(), self.defender_reward())
            #REWARD_MAP[(self.state[self.agents[0]], self.state[self.agents[1]])]

            self.num_moves += 1
            # The dones dictionary must be updated for all players.
            self.dones = {agent: self.num_moves >= NUM_ITERS for agent in self.agents}

            # observe the current state
            for i in self.agents:
                self.observations[i] = self.state[self.agents[self.agent_name_mapping[i]]]
        else:
            # necessary so that observe() returns a reasonable observation at all times.
            ## self.state[self.agents[1 - self.agent_name_mapping[agent]]] = NONE
            # no rewards are allocated until both players give an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()


    def reset_state(self, agent):
        # Reset state per agent
        if agent == "attacker":
            state = np.zeros((self.num_nodes, self.num_nodes))
            state[0][0] = 2 
            state[0][0] = 2 
            state[0, 1:] = 1 
            state[1:, 0] = 1
        elif agent == "defender":
            state = np.ones((self.num_nodes, self.num_nodes))
            np.fill_diagonal(state, 0)
            
            state[self.num_nodes - 1][self.num_nodes - 1] = -1

        return state


    def reset_observation(self, agent):
        # Reset observation per agent
        if agent == "attacker":
            observation = np.zeros((self.num_nodes, self.num_nodes))
            observation[0][0] = 2 
            observation[0, 1:] = 1 
            observation[1:, 0] = 1
        elif agent == "defender":
            observation = np.ones((self.num_nodes, self.num_nodes))
            np.fill_diagonal(observation, 0)
            observation[self.num_nodes - 1][self.num_nodes - 1] = -1

        return observation

    def attacker_reward(self):
        return np.random.randint(-1, 3)

    def defender_reward(self):
        return np.random.randint(-1, 3)

    def state_update(self, agent, state, action):
        if agent == "attacker":
            # choose a random node from list of neighbors
            neighbors = state.diagonal()
            is_compromised = neighbors == 2
            indices = np.arange(neighbors.size)

            target_node = np.random.choice(indices[is_compromised])

            for neighbor in state[target_node]:
                pass
                # if edge exists betwen target node and neighbor
                ## if self.state["defender"]:
                # then action is possible
                #    action_possible = True
                #    break
                #else:
                # remove edge from adjacency matrix
            # if action is possible
                # get exploitability score
                # generate random number
                # if exploitability/ >= random number
                    # apply action to target node


            #state = np.zeros(shape=(self.num_nodes, self.num_nodes), dtype=np.int32)
        elif agent == "defender":
            pass
            #state = np.zeros(shape=(self.num_nodes, self.num_nodes), dtype=np.int32)

        return state

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
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env



from ray.tune.registry import register_env
# import the pettingzoo environment
from ray.rllib.env import PettingZooEnv
from ray import tune
# define how to make the environment. This way takes an optional environment config, num_floors
env_creator = lambda config: env()
# register that way to make the environment under an rllib name
register_env('AutonomousDefenceEnv', lambda config: PettingZooEnv(env_creator(config)))
# now you can use `prison` as an environment
# you can pass arguments to the environment creator with the env_config option in the config


stop = {
    "training_iteration": 150,
    "timesteps_total": 100000,
    "episode_reward_mean": 1000,
}

config = {
    "env": "AutonomousDefenceEnv",
}


results = tune.run("PG", config=config, stop=stop, verbose=1)



