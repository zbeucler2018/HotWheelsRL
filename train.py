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
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from wandb.integration.sb3 import WandbCallback

import wandb
from env_util import GameStates, make_hotwheels_vec_env


class ValidAlgos(Enum):
    PPO = "PPO"
    A2C = "A2C"
    DQN = "DQN"
    # TRPO = "TRPO"


def make_model(_env: Env, _algo: str, _tensorboard_log_path: str) -> PPO | A2C | DQN:
    """
    Returns a model for the give config and env
    """
    _algo = _algo.upper()
    valid_algos = {
        # algo enum, learning rate
        ValidAlgos.PPO.value: (PPO, 0.0003),
        ValidAlgos.A2C.value: (A2C, 0.0007),
        ValidAlgos.DQN.value: (DQN, 0.0001),
    }

    ModelClass, default_learning_rate = valid_algos[_algo]

    model = ModelClass(
        "CnnPolicy",
        _env,
        verbose=1,
        tensorboard_log=_tensorboard_log_path,
        learning_rate=default_learning_rate,
    )

    return model


def main(algorithm, total_training_steps, wandb_api_key, framestack, save_video):
    """
    Trains a HotWheels Agent
    """
    ENV_ID: str = "HotWheelsStuntTrackChallenge-gba"
    LOG_PATH: str = f"{os.getcwd()}/logs"
    MODEL_SAVE_PATH: str = f"{os.getcwd()}/models"
    VIDEO_LENGTH = 1_000
    VIDEO_SAVE_PATH = f"{os.getcwd()}/videos"
    RUN_ID: str | None = None
    MODEL_EVAL_FREQ: int = 150_000

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
        config=_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        reinit=True,
    )
    RUN_ID = _run.id
    wandb_callback = WandbCallback(
        # gradient_save_freq=wandbConfig.gradient_save_freq,
        model_save_path=MODEL_SAVE_PATH,
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

    venv_cls = SubprocVecEnv
    if MAX_ENVS == 1:
        venv_cls = DummyVecEnv

    env = make_hotwheels_vec_env(
        env_id=ENV_ID,
        game_state=GameStates.SINGLE.value,
        n_envs=MAX_ENVS,
        seed=42,
        vec_env_cls=venv_cls,
        # wrapper_class=VecFrameStack if framestack else None,
        # wrapper_kwargs={ 'n_stack': 4 } if framestack else {}
    )

    if framestack:
        env = VecFrameStack(env, n_stack=4)

    if save_video:
        raise NotImplementedError("Saving to video not ready yet")
        # raises an error when attempting to train agent
        # env = VecVideoRecorder(env, VIDEO_SAVE_PATH,
        #                record_video_trigger=lambda x: x == 0, video_length=VIDEO_LENGTH,
        #                name_prefix=f"{RUN_ID}-{ENV_ID}")

    eval_callback = EvalCallback(
        env,
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=MODEL_EVAL_FREQ,
        deterministic=True,
        render=False,
    )

    _callback_list = CallbackList([eval_callback, wandb_callback])

    # make model
    model = make_model(_env=env, _algo=algorithm, _tensorboard_log_path=LOG_PATH)

    # train model
    try:
        model.learn(total_timesteps=total_training_steps, callback=_callback_list)
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
        "--save_video",
        help="Saves a video of the envs after training",
        action="store_true",
    )
    parser.add_argument(
        "--wandb_api_key", help="API key for WandB monitoring", type=str, required=True
    )

    args = parser.parse_args()

    main(
        algorithm=args.algorithm,
        total_training_steps=args.total_training_steps,
        wandb_api_key=args.wandb_api_key,
        framestack=args.framestack,
        save_video=args.save_video,
    )


"""
python train.py --algorithm PPO --total_training_steps 500 --wandb_api_key 

I need to set:
    game_state
    algorithm
    total_training_steps
    wandb_api_key

# TODO: add ability to use these options
optional:
    policy
    learning_rate
    gamma

experimental:
    skip_wandb(_login): bool
    framestack: bool
    grayscale: bool
    number_laps_per_episode: 1 =< int =< 3
    action_space: Action
    max_steps_per_episode: int


"""
