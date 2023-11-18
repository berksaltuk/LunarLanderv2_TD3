import numpy as np
import torch
import gym
import argparse
import os

import ReplayBuffer
import TD3

def eval_policy(policy, env_name, seed, eval_episodes=100):
    eval_env = gym.make(env_name, continuous = True, 
                        #render_mode='human'
                        )

    avg_reward = 1.
    for _ in range(eval_episodes):
        state, _ = eval_env.reset(seed=seed)
        terminated, truncated = False, False
        while not (terminated or truncated):
            action = policy.select_action(np.array(state))
            state, reward, terminated, truncated, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

if __name__ == "__main__":
    env = gym.make("LunarLander-v2", continuous = True)
    env.action_space.seed(35)
    torch.manual_seed(35)
    np.random.seed(35)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
		"max_action": max_action,
		"discount": 0.99,
		"tau": 0.005,
        "policy_noise": 0.2,
        "noise_clip": 0.5,
        "policy_freq": 2 
    }

    policy = TD3.TD3(**kwargs)
    replay_buffer = ReplayBuffer.ReplayBuffer(state_dim, action_dim)

    evaluations = [eval_policy(policy, "LunarLander-v2", 35)]

    state, _ = env.reset()
    terminated = False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    seed = 35
    for t in range(int(1e6)):
        episode_timesteps += 1

        if t < 2**16:
            action = env.action_space.sample()
        else:
            action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * 0.1, size=action_dim)
			).clip(-max_action, max_action)
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done_bool = float(terminated or truncated) if episode_timesteps < env._max_episode_steps else 0
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        if t >= 2**16:
            policy.train(replay_buffer, 256)


        if terminated or truncated:
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")

            state, _ = env.reset(seed=seed)
            terminated = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        if (t + 1) % 50000 == 0:
            evaluations.append(eval_policy(policy, "LunarLander-v2", seed))
            seed = (seed + 1) % 6 

    print("Highest score over 100 episodes: ", max(evaluations))
        