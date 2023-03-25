import argparse
import os
import time
from utils import print_args_table

parser = argparse.ArgumentParser(description='HotWheels training')
parser.add_argument('--algo', type=str, help='the algorithm to use', required=True)
parser.add_argument('--total_timesteps', type=int, help='the total timesteps taken while learning', required=True)
parser.add_argument('--learning_rate', type=float, help='Learning rate for model. ex: 0.000001', required=True)
parser.add_argument('--grayscale', action='store_true', help='whether to use grayscale', default=False)
parser.add_argument('--max_episode_steps', type=int, help='Learning rate for model', default=15_000)
parser.add_argument('--render_mode', type=str, help='render type for observation', default='rgb_array', choices=['rgb_array', 'human'])
parser.add_argument('--policy', type=str, help='policy for model to use', default='CnnPolicy')


args = parser.parse_args()

print_args_table(args)

config = {
    "env_name": "HotWheels",
    "algorithm": args.algo,
    "total_timesteps": args.total_timesteps,
    "learning_rate": args.learning_rate,
    "grayscale": f"{args.grayscale}",
    "max_episode_steps": args.max_episode_steps,
    "policy_type": args.policy,
    "unix_date": time.time()
}



try:
    import retro
    import gymnasium as gym
    from gymnasium.wrappers import GrayScaleObservation, TimeLimit, FrameStack
    from gymnasium.utils.save_video import save_video

    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3 import PPO, A2C
    from sb3_contrib import TRPO

    import wandb
    from wandb.integration.sb3 import WandbCallback

except Exception as e:
    print("Could not import ML libraries")
    raise e

from HotWheelsEnv import FixSpeed, DoTricks, SingleActionEnv, TerminateOnCrash, NorrmalizeBoost


    

env = retro.make("HotWheelsStuntTrackChallenge-gba", render_mode=args.render_mode)
#env = make_env(env)
env = TimeLimit(env, max_episode_steps=args.max_episode_steps)
env = FixSpeed(env)
env = DoTricks(env)
#env = SingleActionEnv(env)
env = TerminateOnCrash(env)
#env = NorrmalizeBoost(env)
if args.grayscale:
    env = GrayScaleObservation(env, keep_dim=True)
# if args.vec:
#     env = make_vec_env(env, n_envs=4)


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



run = wandb.init(
    project="sb3-hotwheels",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    #save_code=True,  # optional (broken)
    #run_name=f"{args.algo}-{wandb.run.id}"
)


try:  
    model = algo(args.policy, env, verbose=1, tensorboard_log=f"{os.getcwd()}/logs", learning_rate=args.learning_rate)
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )
finally:
    env.close()
    run.finish()


model.save(f"{args.algo}")
