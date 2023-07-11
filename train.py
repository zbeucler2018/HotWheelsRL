import argparse
import multiprocessing
import os
from enum import Enum

from gymnasium.core import Env
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
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


def main(algorithm, total_training_steps, wandb_api_key):
    """
    Trains a HotWheels Agent
    """

    # wandb stuff
    os.system(f"wandb login {wandb_api_key}")
    _model_config = {
        "algorithm": algorithm,
        "total_training_steps": total_training_steps,
        "max_steps_per_episode": "no limit",
        "tensorboard_log_path": f"{os.getcwd()}/logs",
    }
    _run = wandb.init(
        project="sb3-hotwheels",
        config=_model_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        reinit=True,
    )
    wandb_callback = WandbCallback(
        # gradient_save_freq=wandbConfig.gradient_save_freq,
        model_save_path=f"models/{_run.id}",
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
        env_id="HotWheelsStuntTrackChallenge-gba",
        game_state=GameStates.SINGLE.value,
        n_envs=MAX_ENVS,
        seed=42,
        vec_env_cls=SubprocVecEnv,
    )

    # make model
    model = make_model(
        _env=env, _algo=algorithm, _tensorboard_log_path=f"{os.getcwd()}/logs"
    )

    # train model
    try:
        model.learn(total_timesteps=total_training_steps, callback=wandb_callback)
    finally:
        env.close()
        del env
        _run.finish()

    # TODO: Record video and save in drive after training is finised. Maybe upload mp4 to wandb
    # record video
    # env = make_hotwheels_vec_env(
    #     env_id="HotWheelsStuntTrackChallenge-gba",
    #     state=game_state,
    #     use_restricted_actions=retro.Actions.DISCRETE,
    #     n_envs=1,
    #     seed=42,
    #     extra_wrappers=_extra_wrappers
    # )


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
        "--wandb_api_key", help="API key for WandB monitoring", type=str, required=True
    )

    args = parser.parse_args()

    main(
        algorithm=args.algorithm,
        total_training_steps=args.total_training_steps,
        wandb_api_key=args.wandb_api_key,
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
    skip_wandb: bool
    framestack: bool
    grayscale: bool
    number_laps_per_episode: 1 =< int =< 3
    action_space: Action
    max_steps_per_episode: int
"""
