import argparse
import os
import time
from utils import print_args_table

parser = argparse.ArgumentParser(description='HotWheels evaluation')
parser.add_argument('--filename', type=str, help='Filename of model to import', required=True)
parser.add_argument('--algo', type=str, help='What algorithm to use (PPO, A2C, TRPO)')
parser.add_argument('--episodes', type=int, help='Total episodes for the agent to run', required=True)
parser.add_argument('--max_episode_steps', type=int, help='The maximum amount of steps per episode when training', default=15_000)
parser.add_argument('--render_mode', type=str, help='render type for observation', default='rgb_array', choices=['rgb_array', 'human'])
parser.add_argument('--record_gif', action='store_true', help='Record a video of the agent', default=False)
parser.add_argument('--grayscale', action='store_true', help='whether to use grayscale', default=False)


args = parser.parse_args()

print_args_table(args)


try:
    import retro
    import imageio
    import numpy as np
    import gymnasium as gym
    from gymnasium.wrappers import GrayScaleObservation, TimeLimit, FrameStack

    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3 import PPO, A2C
    from sb3_contrib import TRPO

except Exception as e:
    print("Could not import ML libraries")
    raise e

class FixSpeed(gym.Wrapper):
    """
    Fixes env bug so the speed is accurate
    """
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info['speed'] *= 0.702
        return observation, reward, terminated, truncated, info
  

class DoTricks(gym.Wrapper):
    """
    Encourages the agent to do tricks (increase score) (+0.1)
    """
    def __init__(self, env, score_boost=1.0):
        super().__init__(env)
        self.prev_score = None
        self.score_boost = score_boost
        
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        # Get the current score and compare it with the previous score
        curr_score = info.get('score')
        if curr_score is not None and self.prev_score is not None:
            if curr_score > self.prev_score:
                reward += (1 / (curr_score - self.prev_score))
        # Update the previous score
        self.prev_score = curr_score
        return observation, reward, terminated, truncated, info
    




env = retro.make("HotWheelsStuntTrackChallenge-gba", render_mode=args.render_mode)
env = TimeLimit(env, max_episode_steps=args.max_episode_steps)
if args.grayscale:
    env = GrayScaleObservation(env, keep_dim=True)
# if args.vec:
#     env = make_vec_env(env, n_envs=4)
#FrameStack



# check if valid env
try:
    check_env(env)
except Exception as err:
    env.close()
    print(err)
    raise





if args.algo == "ppo":
    algo = PPO
elif args.algo == "a2c":
    algo = A2C
elif args.algo == "trpo":
    algo = TRPO
else:
    raise



model = algo.load(args.filename)



for ep in range(args.episodes):
    observation, info = env.reset(seed=42)
    total_reward = 0
    frames = []
    total_steps = 0

    while True:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        
        if args.record_gif:
            frames.append(observation)

        print(f"\n==Episode {ep}==", f"total reward: {total_reward}",f"progress: {info['progress']}", sep="\n")


        if terminated or truncated:
            print(ep, total_reward, info['progress'])
            observation, info = env.reset(seed=42)
            break
    
    if args.record_gif:
        imageio.mimsave(f"{args.algo}-{ep}.gif", [np.array(img) for i, img in enumerate(frames) if i%2 == 0], fps=29)

    

env.close()


