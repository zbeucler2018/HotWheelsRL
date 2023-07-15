import argparse
import multiprocessing
import os
from enum import Enum

import retro
from gymnasium.core import Env
from gymnasium.wrappers import RecordVideo, ResizeObservation
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecVideoRecorder,
)
from wandb.integration.sb3 import WandbCallback

import wandb
from env_util import GameStates, make_hotwheels_vec_env


def run(model_save_path: str, algorithm: str, framestack: bool):
    ENV_ID = "HotWheelsStuntTrackChallenge-gba"
    RUN_DURATION = 3_000

    env = make_hotwheels_vec_env(
        env_id=ENV_ID,
        game_state=GameStates.SINGLE.value,
        n_envs=2,
        seed=42,
        vec_env_cls=SubprocVecEnv,
    )
    if framestack:
        env = VecFrameStack(env, n_stack=4)

    # env = VecVideoRecorder(env, video_folder=f"{os.getcwd()}/videos",
    #     record_video_trigger=lambda x: x == 0, video_length=RUN_DURATION)

    if algorithm.upper() == "PPO":
        model = PPO.load(model_save_path, env=env)
    elif algorithm.upper() == "A2C":
        model = A2C.load(model_save_path, env=env)
    elif algorithm.upper() == "DQN":
        model = DQN.load(model_save_path, env=env)
    else:
        raise Exception(f"Invalid algorithm: {algorithm}")

    vec_env = model.get_env()
    obs = vec_env.reset()

    # sometimes helps to stop terminating
    # on wall hit for full evaluation
    for _ in range(RUN_DURATION):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")  # remove 'human' when recording video

    env.close()


if __name__ == "__main__":
    # run(model_save_path="./model.zip", algorithm="PPO", framestack=True)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--algorithm",
        help="Algorithm to learn",
        type=str,
        required=True,
        choices=["PPO", "A2C", "DQN"],
    )
    parser.add_argument(
        "--model_save_path",
        help="Relative path to model (.zip)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--framestack", help="Uses stacks of 4 frames", action="store_true"
    )

    args = parser.parse_args()

    run(
        algorithm=args.algorithm,
        model_save_path=args.model_save_path,
        framestack=args.framestack,
    )
