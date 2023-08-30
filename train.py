import argparse
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import CallbackList
from callbacks.videoRecorder import VideoRecorderCallback
from callbacks.evalAgent import EvalCallback

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
    DummyVecEnv,
)
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, NormalizeReward
from gym_wrappers import *

import retro

from utils import print_args, in_colab

IN_COLAB = in_colab()
print(f"Running in colab: {IN_COLAB}")


@print_args
def main(
    total_training_steps,
    resume,
    run_id,
    model_path,
    num_envs,
    encourage_speed,
    encourage_tricks,
    crop_obs,
    minimap_obs,
    trim_obs,
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
        _env = retro.make("HotWheelsStuntTrackChallenge-gba", render_mode="rgb_array")
        _env = Monitor(env=_env)
        _env = GrayScaleObservation(_env, keep_dim=True)
        _env = TerminateOnCrash(_env)
        _env = PenalizeHittingWalls(_env)
        _env = FixSpeed(_env)
        _env = HotWheelsDiscretizer(_env)
        if encourage_speed:
            _env = IncreaseMeanSpeed(_env)
        if encourage_tricks:
            _env = EncourageTricks(_env)
        if crop_obs:
            _env = CropObservation(_env)
        if trim_obs:
            _env = NavObservation(_env)

        if minimap_obs:
            _env = MiniMapObservation(_env)
            _env = ResizeObservation(_env, (36, 36))  # resize to something compatible
        else:
            # minimap obs is smaller than 84x84
            _env = ResizeObservation(_env, (84, 84))
        _env = NormalizeReward(_env)
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
    _model_save_path = (
        f"/content/gdrive/MyDrive/HotWheelsRL/data/models/{_run.name}"
        if IN_COLAB
        else f"./models/{_run.name}"
    )
    wandb_callback = WandbCallback(
        # gradient_save_freq=1_000,
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
        eval_freq=max(1_000_000 // num_envs, 1),
        deterministic=True,
        render=False,
    )
    video_callback = VideoRecorderCallback(
        eval_env=venv, render_freq=5_000_000, n_eval_episodes=1, deterministic=True
    )
    _callback_list = CallbackList([eval_callback, wandb_callback, video_callback])

    # setup model
    if resume:
        model = PPO.load(
            path=model_path,
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
        "--encourage_speed",
        help="Give a reward increasing mean speed (0.1 or -0.1)",
        action="store_true",
    )
    parser.add_argument(
        "--encourage_tricks",
        help="Give a reward for doing tricks and increasing score",
        action="store_true",
    )
    parser.add_argument(
        "--crop_obs",
        help="Crop the observation so the model isn't given the entire obs, just a section with the car in it. Smaller than trim_obs",
        action="store_true",
    )
    parser.add_argument(
        "--minimap_obs",
        help="Crop the observation so the model is given only the minimap",
        action="store_true",
    )
    parser.add_argument(
        "--trim_obs",
        help="Crop the observation such that the lap/race timers, speed dial, and minimap are not shown",
        action="store_true",
    )

    args = parser.parse_args()

    main(
        total_training_steps=args.total_steps,
        resume=args.resume,
        run_id=args.run_id,
        model_path=args.model_path,
        num_envs=args.num_envs,
        encourage_speed=args.encourage_speed,
        encourage_tricks=args.encourage_tricks,
        crop_obs=args.crop_obs,
        minimap_obs=args.minimap_obs,
        trim_obs=args.trim_obs,
    )
