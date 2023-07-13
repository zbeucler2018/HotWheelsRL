import argparse
import multiprocessing
import os
from enum import Enum

from gymnasium.core import Env
from gymnasium.wrappers import RecordVideo
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


def resume(
    run_id: str,
    model_save_path: str,
    total_training_steps: int,
    framestack: bool,
    algorithm: str,
    wandb_api_key: str,
):
    """
    Trains a HotWheels Agent
    """
    ENV_ID: str = "HotWheelsStuntTrackChallenge-gba"
    LOG_PATH: str = f"{os.getcwd()}/logs"
    MODEL_SAVE_PATH: str = model_save_path  # f"{os.getcwd()}/models"
    VIDEO_LENGTH = 1_000
    VIDEO_SAVE_PATH = f"videos/"
    RUN_ID: str = run_id

    # wandb stuff
    os.system(f"wandb login {wandb_api_key}")
    _config = {
        "algorithm": algorithm,
        "total_training_steps": total_training_steps,
        "max_steps_per_episode": "no limit",
        "tensorboard_log_path": LOG_PATH,  # is this needed?
        "framestack": framestack,
    }
    _run = wandb.init(
        project="sb3-hotwheels",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        id=RUN_ID,
        resume=True,
    )
    wandb_callback = WandbCallback(
        # gradient_save_freq=wandbConfig.gradient_save_freq,
        model_save_path=f"{os.getcwd()}/models",
        model_save_freq=25_000,
        verbose=1,
    )
    # TODO: Figure out how to wandb log info stuff in vec envs (https://github.com/wandb/wandb/issues/5087)
    # hw_callback = HotWheelsCallback(
    #     _model_save_path=f"models/{_run.id}",
    #     _model_save_freq=25_000,
    #    # verbose=1
    # )

    MAX_ENVS = multiprocessing.cpu_count()
    print(f"Using {MAX_ENVS} CPUs")

    env = make_hotwheels_vec_env(
        env_id=ENV_ID,
        game_state=GameStates.SINGLE.value,
        n_envs=MAX_ENVS,
        seed=42,
        vec_env_cls=SubprocVecEnv,
        # wrapper_class=VecFrameStack if framestack else None,
        # wrapper_kwargs={ 'n_stack': 4 } if framestack else {}
    )

    if framestack:
        env = VecFrameStack(env, n_stack=4)

    # load model
    if algorithm.upper() == "PPO":
        model = PPO.load(MODEL_SAVE_PATH, env=env)
    elif algorithm.upper() == "A2C":
        model = A2C.load(MODEL_SAVE_PATH, env=env)
    elif algorithm.upper() == "DQN":
        model = DQN.load(MODEL_SAVE_PATH, env=env)
    else:
        raise Exception(f"Invalid algorithm: {algorithm}")

    # train model
    try:
        model.learn(total_timesteps=total_training_steps, callback=wandb_callback)
    finally:
        env.close()
        del env
        _run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--algorithm",
        help="Algorithm to learn",
        type=str,
        required=True,
        choices=["PPO", "A2C", "DQN"],
    )
    parser.add_argument(
        "--total_training_steps", help="Total steps to train", type=int, required=True
    )
    parser.add_argument(
        "--framestack", help="Uses stacks of 4 frames", action="store_true"
    )
    parser.add_argument(
        "--wandb_api_key", help="API key for WandB monitoring", type=str, required=True
    )
    parser.add_argument(
        "--run_id",
        help="Wandb run ID for the particular model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model_save_path",
        help="Relative path to model (.zip)",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    resume(
        algorithm=args.algorithm,
        total_training_steps=args.total_training_steps,
        wandb_api_key=args.wandb_api_key,
        framestack=args.framestack,
        run_id=args.run_id,
        model_save_path=args.model_save_path,
    )
