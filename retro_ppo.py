"""
Train an agent using Proximal Policy Optimization from Stable Baselines 3

Taken from: https://github.com/Farama-Foundation/stable-retro/blob/master/retro/examples/ppo.py
"""

import argparse

import gymnasium as gym
import numpy as np
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)

from utils import print_args, in_colab

IN_COLAB = in_colab()
print(f"Running in colab: {IN_COLAB}")

import retro

from gym_wrappers import *

from callbacks.evalAgent import EvalCallback

import wandb
from wandb.integration.sb3 import WandbCallback


def make_retro(
    *, game, state=None, max_episode_steps=4500, render_mode="rgb_array", **kwargs
):
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game, state, render_mode=render_mode, **kwargs)
    env = Monitor(env)
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    env = TerminateOnCrash(env)
    env = PenalizeHittingWalls(env)
    env = HotWheelsDiscretizer(env)

    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def wrap_deepmind_retro(env):
    """
    Configure environment for retro games, using config similar to DeepMind-style Atari in openai/baseline's wrap_deepmind
    """
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="HotWheelsStuntTrackChallenge-gba")
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--scenario", default=None)
    parser.add_argument(
        "--total_steps", help="Total steps to train", type=int, required=True
    )
    parser.add_argument("--resume", help="Resume training a model", action="store_true")
    parser.add_argument(
        "--run_id", help="Wandb run ID to resume training a model", type=str
    )
    parser.add_argument(
        "--model_path", help="Path to saved model to resume training", type=str
    )
    parser.add_argument(
        "--num_envs",
        help="Number of envs to train at the same time. Default is 8",
        type=int,
        required=False,
        default=8,
    )

    args = parser.parse_args()

    def make_env():
        env = make_retro(game=args.game, state=args.state, scenario=args.scenario)
        env = wrap_deepmind_retro(env)
        return env

    venv = VecTransposeImage(
        VecFrameStack(SubprocVecEnv([make_env] * args.num_envs), n_stack=4)
    )
    if args.resume:
        model = PPO.load(
            path=args.model_path,
            env=venv,
            # Needed because sometimes sb3 cant find the
            # obs and action space. Seen in colab on 8/21/23
            custom_objects={
                "observation_space": venv.observation_space,
                "action_space": venv.action_space,
            },
        )
    else:
        model = PPO(
            policy="CnnPolicy",
            env=venv,
            learning_rate=lambda f: f * 2.5e-4,
            n_steps=128,
            batch_size=32,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log="./logs/",
        )

    # setup wandb
    _config = {
        "algorithm": "PPO",
        "total_training_steps": args.total_steps,
        "framestack": True,
    }
    _run = wandb.init(
        project="sb3-hotwheels",
        config=_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        resume=True if args.resume else None,
        id=args.run_id if args.run_id else None,
    )
    # setup callbacks
    _model_save_path = (
        f"/content/gdrive/MyDrive/HotWheelsRL/data/models/{_run.name}"
        if IN_COLAB
        else f"./models/{_run.name}"
    )
    wandb_callback = WandbCallback(
        gradient_save_freq=10_000,
        model_save_path=_model_save_path,
        model_save_freq=50_000,
        verbose=1,
    )
    _best_model_save_path = (
        f"/content/gdrive/MyDrive/HotWheelsRL/data/best_models/{_run.name}"
        if IN_COLAB
        else f"./best_models/{_run.name}"
    )
    eval_callback = EvalCallback(
        venv,
        best_model_save_path=_best_model_save_path,
        log_path=f"./logs/{_run.name}",
        eval_freq=max(100_000 // args.num_envs, 1),
        deterministic=True,
        render=False,
    )
    _callback_list = CallbackList([eval_callback, wandb_callback])

    model.learn(
        total_timesteps=args.total_steps,
        log_interval=1,
        callback=_callback_list,
    )


if __name__ == "__main__":
    main()
