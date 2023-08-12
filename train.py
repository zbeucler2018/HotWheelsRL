import argparse
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
    DummyVecEnv,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import ClipRewardEnv
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation
from gym_wrappers import *

import retro

from utils import print_args


@print_args
def main(
    total_training_steps,
    resume,
    run_id,
    model_path,
    num_envs,
    encourage_tricks,
    crop_obs,
    minimap_obs,
) -> None:
    # check if we want to resume
    if resume or run_id:
        assert (
            resume
        ), "--resume, --run_id, and --model_path must be populated to resume training a model"
        assert (
            run_id
        ), "--resume, --run_id, and --model_path must be populated to resume training a model"
        assert (
            model_path
        ), "--resume, --run_id, and --model_path must be populated to resume training a model"

    assert not (minimap_obs and crop_obs), "--minimap_obs or --crop_obs, not both"

    def make_retro():
        _env = retro.make(
        "HotWheelsStuntTrackChallenge-gba", render_mode="rgb_array"
        )
        _env = Monitor(env=_env)
        _env = GrayScaleObservation(_env, keep_dim=True)
        _env = TerminateOnCrash(_env)
        _env = FixSpeed(_env)
        _env = HotWheelsDiscretizer(_env)
        _env = ClipRewardEnv(_env)
        if encourage_tricks:
            _env = EncourageTricks(_env)
        if crop_obs:
            _env = CropObservation(_env)
        if minimap_obs:
            _env = MiniMapObservation(_env)
            _env = ResizeObservation(_env, (36, 36))  # resize to something compatible
        else:
            # minimap obs is smaller than 84x84
            _env = ResizeObservation(_env, (84, 84))
        return _env



    # create env
    if num_envs == 1:
        venv = VecTransposeImage(VecFrameStack(DummyVecEnv([make_retro]), n_stack=4))
    else:
        venv = VecTransposeImage(
            VecFrameStack(SubprocVecEnv([make_retro] * num_envs), n_stack=4)
        )

    # setup wandb
    _config = {
        "algorithm": "PPO",
        "total_training_steps": total_training_steps,
        "framestack": True,
    }
    _run = wandb.init(
        project="sb3-hotwheels",
        config=_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        resume=True if resume else None,
        id=run_id if run_id else None,
    )

    # setup callbacks
    wandb_callback = WandbCallback(
        # gradient_save_freq=1_000,
        model_save_path=f"./models/{_run.name}",
        model_save_freq=50_000,
        verbose=1,
    )
    eval_callback = EvalCallback(
        venv,
        best_model_save_path=f"./best_model/{_run.name}/",
        log_path="./logs/",
        eval_freq=150_000,
        deterministic=True,
        render=False,
    )
    _callback_list = CallbackList([eval_callback, wandb_callback])

    # setup model
    if resume:
        model = PPO.load(path=model_path, env=venv)
    else:
        model = PPO(
            "CnnPolicy",
            venv,
            verbose=1,
            tensorboard_log="./logs/",
            # from https://arxiv.org/pdf/1707.06347.pdf
            learning_rate=2.5e-4,
            n_steps=128,
            n_epochs=3,
            batch_size=32,
            ent_coef=0.01,
            vf_coef=1.0,
        )
    try:
        model.learn(
            total_timesteps=total_training_steps,
            callback=_callback_list,
            reset_num_timesteps=False if resume else True,
        )
    finally:
        venv.close()
        _run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--total_steps", help="Total steps to train", type=int, required=True
    )
    parser.add_argument("--resume", help="Resume training a model", action="store_true")
    parser.add_argument(
        "--run_id", help="Wandb run ID to resume training a model", type=str
    )
    parser.add_argument(
        "--model_path", help="Path to saved model to resume training", type=str
    )
    parser.add_argument(
        "--num_envs",
        help="Number of envs to train at the same time. Default is 8",
        type=int,
        required=False,
        default=8,
    )
    parser.add_argument(
        "--encourage_tricks",
        help="Give a reward for doing tricks and increasing score",
        action="store_true",
    )
    parser.add_argument(
        "--crop_obs",
        help="Crop the observation so the model isn't given the entire obs, just a section with the car in it",
        action="store_true",
    )
    parser.add_argument(
        "--minimap_obs",
        help="Crop the observation so the model is given only the minimap",
        action="store_true",
    )

    args = parser.parse_args()

    main(
        total_training_steps=args.total_steps,
        resume=args.resume,
        run_id=args.run_id,
        model_path=args.model_path,
        num_envs=args.num_envs,
        encourage_tricks=args.encourage_tricks,
        crop_obs=args.crop_obs,
        minimap_obs=args.minimap_obs,
    )
