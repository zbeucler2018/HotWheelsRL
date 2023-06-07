import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import wandb
from gymnasium.core import Env
from gymnasium.wrappers import TimeLimit
#from sb3_contrib import TRPO
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.env_checker import check_env
from wandb.integration.sb3 import WandbCallback

from HotWheelsEnv import HotWheelsEnv


class ValidAlgos(Enum):
    PPO = "ppo"
    A2C = "A2C"
    DQN = "DQN"
    #TRPO = "TRPO"


@dataclass
class WandbConfig:
    """ 
    Config for WandB Monitoring.
        model_save_freq: Timestep(?) frequency to save the model
        model_save_path: Filepath to save the trained model
        verbose: Level of verbosity
        gradient_save_freq: ???
    """
    model_save_freq: int
    model_save_path: Union[str, None]
    hot_wheels_env_type: HotWheelsEnv
    project_key: str = "sb3-hotwheels"
    verbose: int = 2
    #gradient_save_freq: int = 100 # TODO: Figure what what this dos and what data type it needs



@dataclass
class ModelConfig:
    """
    Config for the model being trained
        policy: Model policy
        total_training_timesteps: Total timesteps to train model
        learning_rate: Learning rate for model
        max_episode_steps: Max steps an agent can take in an episode (TimeLimit)
        gamma: Discount factor (encourage shorter episodes)
    """
    policy: str
    total_training_timesteps: int
    tensorboard_log_path: str = f"{os.getcwd()}/logs"
    learning_rate: Union[float, int] = 0.000001
    max_episode_steps: int = 25_000
    gamma: float = 0.99




class Trainer:
    """ Trains an agent """

    def __init__(self, env: Union[Env]) -> None:
        try:
            check_env(env, skip_render_check=True)
        except Exception as err:
            print(f"Cannot train agent. Enviroment is invalid")
            env.close()
            raise err
        self.env = env



    def train(self, algo: ValidAlgos, modelConfig: ModelConfig, wandbConfig: WandbConfig) -> None:
        """ Trains an agent using the given configs and algo """

        # validate algo
        if algo not in ValidAlgos:
            raise Exception(f"{algo.value} is not a valid algo")
        
        # set up wandb monitoring
        _run = wandb.init(
                project="sb3-hotwheels",
                config=modelConfig,
                sync_tensorboard=True # auto-upload sb3's tensorboard metrics
            )
        
        # add a TimeLimit
        env = TimeLimit(env, max_episode_steps=modelConfig.max_episode_steps)

        # make model
        if algo == ValidAlgos.PPO:
            model = PPO(
                modelConfig.policy, 
                self.env, 
                verbose=1, 
                tensorboard_log=modelConfig.tensorboard_log_path, 
                learning_rate=modelConfig.learning_rate,
                gamma=modelConfig.gamma
            )
        elif algo == ValidAlgos.A2C:
            model = A2C(
                modelConfig.policy,
                self.env,
                verbose=1,
                tensorboard_log=modelConfig.tensorboard_log_path, 
                learning_rate=modelConfig.learning_rate,
                gamma=modelConfig.gamma
            )
        elif algo == ValidAlgos.DQN:
            model = DQN(
                modelConfig.policy,
                self.env,
                verbose=1,
                tensorboard_log=modelConfig.tensorboard_log_path, 
                learning_rate=modelConfig.learning_rate,
                gamma=modelConfig.gamma
            )
        elif algo == ValidAlgos.TRPO:
            raise NotImplementedError(f"TRPO not setup yet")
        else:
            raise Exception(f"Trying to train a invalid algo")

        # train model
        try:  
            model.learn(
                total_timesteps=modelConfig.total_training_timesteps,
                progress_bar=False, # try this out
                callback=WandbCallback(
                    #gradient_save_freq=wandbConfig.gradient_save_freq,
                    model_save_path=f"models/{_run.id}" if not wandbConfig.model_save_path else wandbConfig.model_save_path,
                    model_save_freq=wandbConfig.model_save_freq,
                    verbose=wandbConfig.verbose
                ),
            )
        # except Exception as err:
        #     self.env.close()
        #     raise err
        finally:
            self.env.close()
            _run.finish()


    def resume_training(self):
        """ Resumes the training of a model """
        raise NotImplementedError(f"Method isnt written yet")