#Create new Trainer and restore its state from the last checkpoint.
import os
import pickle

from ray.rllib.agents.registry import get_trainer_class
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env import PettingZooEnv
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env
from gym.wrappers.monitoring.video_recorder import VideoRecorder

import numpy as np
import tensorflow as tf
import pandas as pd

from rl_autonomous_defence import utils
from rl_autonomous_defence import ade
from rl_autonomous_defence import policy_config
from rl_autonomous_defence.action_mask_model import ActionMaskModel
from rl_autonomous_defence.serve_client import get_action, DEFENDER_URL, ATTACKER_URL


register_env('AutonomousDefenceEnv', lambda x: PettingZooEnv(ade.env()))

env = PettingZooEnv(ade.env())
unwrapped_env = env.env.unwrapped

config = {
    "env": "AutonomousDefenceEnv",
}

serve_urls ={"attacker": ATTACKER_URL, "defender": DEFENDER_URL}

PROJECT_ROOT = "/Users/robertcampbell/Documents/Schoolwork/299/RL-for-Autonomous-Network-Defense-in-SDN/ray_results"
EXPERIMENT_NAME = "test"
PARAM_LIST = "topology=linear_timesteps=100000_reward=gabirondo_lr=0.0004_winthresh=0"
RUN_FOLDER = "CheckpointWrapperPPO_2022-03-14_00-41-05/CheckpointWrapperPPO_AutonomousDefenceEnv_f5adc_00000_0_2022-03-14_00-41-05"

env = env.env.unwrapped 

env.metadata["render.modes"].append("rgb_array")
env.metadata["render_fps"] = 1

rec = VideoRecorder(env, base_path="./video")
env.reset()
#obs = env.observe("attacker")


num_episodes = 0
num_episodes_during_inference = 10
episode_reward = 0.0
step = 0

obs = env.observe("attacker")

episode_rewards = {"attacker": [], "defender": [], "draw": []}
episode_actions = {"attacker": [], "defender": []}

policy_mapping = {0: "attacker", 1: "defender"}
agent_mapping = {"attacker": "attacker", "defender": "defender_v0"}


actions_df = pd.DataFrame(columns=["agent", "step", "action_0", "action_1", "episode"])
rewards_df = pd.DataFrame(columns=["episode", "agent", "reward"])

while num_episodes < num_episodes_during_inference:
    # Compute an action (`a`).
    agent = policy_mapping[step % 2]

    #print(obs)

    flattened_observation = obs["observation"].flatten()
    action_mask = obs["action_mask"]

    #print(obs["observation"])
    #print(agent, action_mask)

    obs = np.concatenate([action_mask, flattened_observation])

    #observation = tf.constant(obs, shape=(1, 75),dtype=np.int64)
    #timestep = tf.constant(step, dtype=tf.int64)

    action_0, action_1 = get_action(serve_urls[agent], obs.tolist(), step)

    env.step((action_0, action_1))

    next_agent = policy_mapping[(step + 1) % 2]

    obs, rewards, dones = env.observe(next_agent), env.rewards, env.dones



    #if os.getenv("RL_SDN_ACTIONSPACE") == "multi":
    #    action_0 = model_output["actions_0"].numpy()
    #    action_1 = model_output["actions_1"][0].numpy()
    #else:
    #    action_0 = model_output["actions"][0].numpy() % 8
    #    action_1 = model_output["actions"][0].numpy() // 8


    #action = trainer.compute_single_action(
    #    observation=obs[agent],
    #    explore=True,
    #    policy_id=agent,  # <- default value
    #)

    """action1 = int(input("Target node: "))
    action2 = int(input("Target action: "))
    action = action1 + action2 * 8"""
    a = {agent: (action_0, action_1)}
    # print(f"Agent: {agent} {action % 8} {action // 8}")
    # Send the computed action `a` to the env.

    #print(obs, rewards, dones)

    #print(obs, rewards, dones)

    rec.capture_frame()

    row = {
        "agent": agent,
        "step": step,
        "action_0": action_0, #% 8,
        "action_1": action_1, #// 8,
        "episode": num_episodes
    }
    actions_df = actions_df.append(row, ignore_index=True)

    step += 1

    #episode_reward += rewards
    # Is the episode `done`? -> Reset.

    #print(obs, rewards, dones)
    if dones[agent]:
        print(f"Episode done: Total reward = {rewards[agent]}")
    # rec = VideoRecorder(env, base_path="/Users/robertcampbell/Documents/Schoolwork/299/RL-for-Autonomous-Network-Defense-in-SDN/ray_results/move_flag_explore_topo/stdis=0.5_stdes=0.5_moveflag=False_exploretopo=False_icm=True/PPO/PPO_AutonomousDefenceEnv_1b7b5_00000_0_2022-02-01_19-14-37")
        #print(f"Agent = {agent}")
        if rewards[agent] == 0:
            episode_rewards["draw"].append(0)
            row = {
                "episode": num_episodes,
                "agent": "draw",
                "reward": 0
            }
            rewards_df = rewards_df.append(row, ignore_index=True)
            #input()
        else:
            episode_rewards[agent].append(rewards[agent])
            row = {
                "episode": num_episodes,
                "agent": agent,
                "reward": rewards[agent]
            }
            rewards_df = rewards_df.append(row, ignore_index=True)
        env.reset()
        obs = env.observe("attacker")

        num_episodes += 1
        episode_reward = 0.0
        step = 0

rec.close()

defender_win_percentage = len(episode_rewards["defender"]) / num_episodes_during_inference
attacker_win_percentage = len(episode_rewards["attacker"]) / num_episodes_during_inference
draw_percentage = len(episode_rewards["draw"]) / num_episodes_during_inference

attacker_rewards = sum(episode_rewards["attacker"])
defender_rewards = sum(episode_rewards["defender"])

mean_attacker_reward = (attacker_rewards - defender_rewards) / num_episodes_during_inference
mean_defender_reward = (defender_rewards - attacker_rewards) / num_episodes_during_inference

print(f"Defender WP - {defender_win_percentage}")
print(f"Attacker WP - {attacker_win_percentage}")
print(f"Draw percentage - {draw_percentage}")

print(f"Attacker average reward = {mean_attacker_reward}")
print(f"Defender average reward = {mean_defender_reward}")

actions_df.to_csv("actions_df.csv", index=False)
rewards_df.to_csv("rewards_df.csv", index=False)
