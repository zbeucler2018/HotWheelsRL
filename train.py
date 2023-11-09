"""
Train an agent using Proximal Policy Optimization from Stable Baselines 3

Taken from: https://github.com/Farama-Foundation/stable-retro/blob/master/retro/examples/ppo.py
"""

import argparse
from train_utils import make_retro, wrap_deepmind_retro
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)
from gym_wrappers import *
from callbacks.evalAgent import EvalCallback
import wandb
from wandb.integration.sb3 import WandbCallback
from utils import in_colab, parse_args, print_args


#@print_args
def main(args) -> None:
    ef = max(5_000 // args.num_envs, 1)  # max(args.num_steps // args.num_envs, 1)
    print(f"Eval freq: {ef}")
    IN_COLAB = in_colab()
    print(f"Running in colab: {IN_COLAB}")

    def make_env():
        env = make_retro(game=args.game, state=args.state, scenario=args.scenario)
        env = wrap_deepmind_retro(env)
        env = HotWheelsWrapper(env) # allows us to change to eval state
        return env

    venv = VecTransposeImage(
        VecFrameStack(SubprocVecEnv([make_env] * args.num_envs), n_stack=4)
    )

    # setup wandb
    _run = wandb.init(
        project="sb3-hotwheels",
        config={
            "algorithm": "PPO",
            "total_training_steps": args.total_steps,
            "framestack": True,
        },
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        resume=True if args.resume else None,
        id=args.run_id if args.run_id else None,
        tags=[args.state],
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
            tensorboard_log=f"./logs/{_run.name}",
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
        eval_freq=ef,
        deterministic=True,
        render=True,
    )
    _callback_list = CallbackList([eval_callback, wandb_callback])

    try:
        model.learn(
            total_timesteps=args.total_steps,
            log_interval=1,
            callback=_callback_list,
            reset_num_timesteps=False if args.resume else True,
        )
    finally:
        venv.close()
        _run.finish()


if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #args = parse_args(parser)

    from dataclasses import dataclass
    from utils import HotWheelsStates

    ### MOSTLY FOR DEBUGGING

    @dataclass
    class A:
        total_steps: int
        num_envs: int
        game: str = "HotWheelsStuntTrackChallenge-gba"
        state: HotWheelsStates = HotWheelsStates.TREX_VALLEY_SINGLE
        scenario: any = None
        resume: bool = False
        run_id: any = None


    args = A(
        total_steps=5000,
        num_envs=4
    )

    main(args)
