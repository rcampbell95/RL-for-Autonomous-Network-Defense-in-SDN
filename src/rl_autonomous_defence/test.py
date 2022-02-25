#Create new Trainer and restore its state from the last checkpoint.

from ray.rllib.agents.registry import get_trainer_class
from ray.rllib.env import PettingZooEnv

from ray.tune.registry import register_env

import os

import ade
import policy_config
import pandas as pd

import utils
import os

import tensorflow as tf

from gym.wrappers.monitoring.video_recorder import VideoRecorder


ALGO = "PPO"
CHECKPOINT1 = "/Users/robertcampbell/Documents/Schoolwork/299/RL-for-Autonomous-Network-Defense-in-SDN/ray_results/patch_defender_actions/stdis=0.5_stdes=0.5_moveflag=False_exploretopo=False_isolatenode=True_reward-costs-clipped=True/PPO/PPO_AutonomousDefenceEnv_2bed3_00000_0_2022-02-24_18-47-48/checkpoint_000012/checkpoint-12"
CHECKPOINT2 = "/Users/robertcampbell/Documents/Schoolwork/299/RL-for-Autonomous-Network-Defense-in-SDN/ray_results/patch_defender_actions/stdis=0.5_stdes=0.5_moveflag=False_exploretopo=False_isolatenode=True_reward-costs-clipped=True/PPO/PPO_AutonomousDefenceEnv_2bed3_00000_0_2022-02-24_18-47-48/checkpoint_000012/checkpoint-12"


register_env('AutonomousDefenceEnv', lambda x: PettingZooEnv(ade.env()))

trainer1 = get_trainer_class(ALGO)(config=policy_config.config)
trainer2 = get_trainer_class(ALGO)(config=policy_config.config)
trainer1.restore(CHECKPOINT1)
trainer2.restore(CHECKPOINT2)

trainers = [trainer1, trainer2]

policy = trainer1.get_policy("attacker")
#policy.model.base_model.summary()

# Create the env to do inference in.

env = PettingZooEnv(ade.env())
env.metadata["render.modes"].append("rgb_array")

rec = VideoRecorder(env.env.unwrapped, base_path="./videos")
obs = env.reset()

num_episodes = 0
num_episodes_during_inference = 1
episode_reward = 0.0
step = 0

episode_rewards = {"attacker": [], "defender": [], "draw": []}
episode_actions = {"attacker": [], "defender": []}

policy_mapping = {0: "attacker", 1: "defender"}

actions_df = pd.DataFrame(columns=["agent", "step", "action_0", "action_1", "episode"])
rewards_df = pd.DataFrame(columns=["episode", "agent", "reward"])

while num_episodes < num_episodes_during_inference:
    # Compute an action (`a`).
    trainer = trainers[step % 2]

    agent = policy_mapping[step % 2]
    action = trainer.compute_single_action(
        observation=obs[agent],
        explore=True,
        policy_id=agent,  # <- default value
    )

    """action1 = int(input("Target node: "))
    action2 = int(input("Target action: "))
    action = action1 + action2 * 8"""
    a = {agent: action}
    # print(f"Agent: {agent} {action % 8} {action // 8}")
    # Send the computed action `a` to the env.
    obs, rewards, dones, _ = env.step(a)

    rec.capture_frame()

    if os.getenv("RL_SDN_AUTOREG") == "True":
        action_0 = action[0]
        action_1 = action[1]
    else:
        action_0 = action % 8
        action_1 = action // 8

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
    if dones[policy_mapping[step % 2]]:
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
        obs = env.reset()
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
