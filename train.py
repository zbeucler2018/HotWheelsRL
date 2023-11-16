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
from wrappers.hotwheels import HotWheelsWrapper
from callbacks.evalAgent import EvalCallback
import wandb
from wandb.integration.sb3 import WandbCallback
from utils import in_colab, parse_args, print_args


# @print_args
def main(args) -> None:
    ef = max(100_000 // args.num_envs, 1)  # max(args.num_steps // args.num_envs, 1)
    print(f"Eval freq: {ef}")
    IN_COLAB = in_colab()
    print(f"Running in colab: {IN_COLAB}")

    def make_env():
        env = make_retro(game=args.game, state=args.state, scenario=args.scenario)
        env = wrap_deepmind_retro(env)
        env = HotWheelsWrapper(env)  # allows us to change to eval state
        return env

    venv = VecTransposeImage(
        VecFrameStack(SubprocVecEnv([make_env] * args.num_envs), n_stack=4)
    )

    if args.training_states:
        # Need to change state AFTER adding SubProcVec because
        # retro will throw "1 Emulator per process only" exception
        # if applied before
        for indx, t_state in enumerate(args.training_states):
            _ = venv.env_method(
                method_name="load_state", statename=f"{t_state}.state", indices=indx
            )
            _ = venv.env_method(method_name="reset_emulator_data", indices=indx)
        _ = venv.reset()

    # setup wandb
    _run = wandb.init(
        project="sb3-hotwheels",
        config=args,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        resume=True if args.resume else None,
        id=args.run_id if args.run_id else None,
        tags=[args.state],
        dir="./logs/wandb/",
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
            tensorboard_log=f"./logs/tf/{_run.name}",
        )

    # setup callbacks
    _model_save_path = (
        f"/content/gdrive/MyDrive/HotWheelsRL/data/models/{_run.name}"
        if IN_COLAB
        else f"./models/models/{_run.name}"
    )
    wandb_callback = WandbCallback(
        gradient_save_freq=50_000,
        model_save_path=_model_save_path,
        model_save_freq=50_000,
        verbose=1,
    )
    _best_model_save_path = (
        f"/content/gdrive/MyDrive/HotWheelsRL/data/best_models/{_run.name}"
        if IN_COLAB
        else f"./models/best_models/{_run.name}"
    )

    eval_callback = EvalCallback(
        venv,
        best_model_save_path=_best_model_save_path,
        log_path=f"./logs/eval/{_run.name}",
        eval_freq=ef,
        eval_statename=args.evaluation_state,
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
    parser = argparse.ArgumentParser()
    args = parse_args(parser)
    print(args)

    main(args)
