"""
Train an agent using Proximal Policy Optimization from Stable Baselines 3

Taken from: https://github.com/Farama-Foundation/stable-retro/blob/master/retro/examples/ppo.py
"""

import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)
from wrappers.hotwheels import HotWheelsWrapper
from evaluation.evalCallback import EvalCallback
import wandb
from wandb.integration.sb3 import WandbCallback
from utils import Config, make_retro


def main(config: Config) -> None:
    def make_env():
        env = make_retro(game=config.game, state=config.state)
        env = HotWheelsWrapper(
            env,
            action_space=config.action_space,
            frame_skip=config.frame_skip,
            frame_skip_stickprob=config.frame_skip_prob,
            terminate_on_crash=config.terminate_on_crash,
            terminate_on_wall_crash=config.terminate_on_wall_crash,
            crash_reward=config.crash_reward,
            wall_crash_reward=config.wall_crash_reward,
            use_deepmind_wrapper=True,
            max_episode_steps=5_100,
        )
        return env

    venv = VecTransposeImage(
        VecFrameStack(
            SubprocVecEnv([make_env] * config.num_envs), n_stack=config.frame_stack
        )
    )

    if config.training_states:
        # Need to change state AFTER adding SubProcVec because
        # retro will throw "1 Emulator per process only" exception
        # if applied before
        for indx, t_state in enumerate(config.training_states):
            _ = venv.env_method(
                method_name="load_state", statename=f"{t_state}.state", indices=indx
            )
            _ = venv.env_method(method_name="reset_emulator_data", indices=indx)
        _ = venv.reset()

    # setup wandb monitoring
    _run = wandb.init(
        project="sb3-hotwheels",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        resume=True if config.resume else None,
        id=config.run_id if config.run_id else None,
        tags=[config.state],
        dir="./logs/wandb/",
    )

    if config.resume:
        model = PPO.load(
            path=config.model_load_path,
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
            policy=config.policy,
            env=venv,
            learning_rate=lambda f: f * 2.5e-4,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            verbose=1,
            tensorboard_log=f"./logs/tf/{_run.name}",
        )

    # setup callbacks
    _model_save_path = (
        config.gdrive_model_save_path if config.in_colab else config.model_save_path
    )
    wandb_callback = WandbCallback(
        # gradient_save_freq=50_000,
        model_save_path=_model_save_path + _run.name,
        model_save_freq=config.model_save_freq,
        verbose=1,
    )

    _best_model_save_path = (
        config.gdrive_best_model_save_path
        if config.in_colab
        else config.best_model_save_path
    )
    eval_callback = EvalCallback(
        venv,
        best_model_save_path=_best_model_save_path + _run.name,
        log_path=f"./logs/eval/{_run.name}",
        eval_freq=config.eval_freq,
        eval_statename=config.evaluation_statename,
        deterministic=True,
        render=config.render_eval,
    )
    _callback_list = CallbackList([eval_callback, wandb_callback])

    try:
        model.learn(
            total_timesteps=config.total_steps,
            log_interval=1,
            callback=_callback_list,
            reset_num_timesteps=False if config.resume else True,
        )
    finally:
        venv.close()
        _run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Filepath to config yaml file", required=False)
    args = parser.parse_args()
    _config = Config(args.config)
    print(_config)
    main(_config)
