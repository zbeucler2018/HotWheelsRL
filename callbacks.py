import pprint

import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from wandb.integration.sb3 import WandbCallback

import wandb

# log to wandb

# run agent and record video?


class HotWheelsCallback(WandbCallback):
    def __init__(verbose, _model_save_path, _model_save_freq):
        super().__init__(
            verbose=1,
            model_save_path=_model_save_path,
            model_save_freq=_model_save_freq,
        )

    def _on_step(self) -> bool:
        # print(self.locals['infos'])
        for cpu in self.locals["infos"]:
            print(cpu)
            wandb.log(cpu)
        # wandb.log(self.locals['infos'][0])
        return super()._on_step()
