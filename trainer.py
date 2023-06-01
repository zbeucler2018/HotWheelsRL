from typing import Union, Optional
import os

from gymnasium.core import Env

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, A2C
from sb3_contrib import TRPO

import wandb
from wandb.integration.sb3 import WandbCallback

from HotWheelsEnv import HotWheelsEnv

from enum import Enum

from dataclasses import dataclass


class ValidAlgos(Enum):
    PPO = "ppo"
    A2C = "A2C"
    TRPO = "TRPO"






"""
input: env, config
1. env_check
2. train with algo
    a. set up wandb
    b. set up model config
    c. set up train config
    e. 

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
"""


@dataclass
class WandbConfig:
    api_key: str
    project_key: str = "sb3-hotwheels"
    gradient_save_freq: int = 100
    model_save_path: str
    model_save_freq: int
    verbose: int


@dataclass
class ModelConfig:
    policy: str
    tensorboard_log_path: str = f"{os.getcwd()}/logs"
    learning_rate: Union[float, int] = 0.000001
    total_timesteps: int
    gamma: float = 0.99




class Trainer:
    """ Trains an agent """

    def __init__(self, env: Env) -> None:
        try:
            check_env(env, skip_render_check=True)
        except Exception as err:
            print(f"Cannot train agent. Enviroment is invalid")
            env.close()
            raise err
        
        self.env = env



    def train(self, algo: ValidAlgos, modelConfig: ModelConfig, wandbConfig: WandbConfig) -> None:

        if algo not in ValidAlgos:
            raise Exception(f"{algo.value} is not a valid algo")
        
        _run = wandb.init(
                project="sb3-hotwheels",
                config=config,
                sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                monitor_gym=True  # auto-upload the videos of agents playing the game
                #save_code=True,  # optional (broken)
                #run_name=f"{args.algo}-{wandb.run.id}"
            )


        pass