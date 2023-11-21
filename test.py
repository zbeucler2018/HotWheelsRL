from callbacks.eval_policy import evaluate_policy

# from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
    VecVideoRecorder,
    DummyVecEnv,
)
from wrappers.viewer import Viewer
from wrappers.hotwheels import HotWheelsWrapper
import numpy as np
from utils import HotWheelsStates, make_retro


def make_env():
    env = make_retro(
        game="HotWheelsStuntTrackChallenge-gba",
        state=HotWheelsStates.DINO_BONEYARD_MULTI,
    )
    env = HotWheelsWrapper(env)
    return env


venv = VecTransposeImage(VecFrameStack(DummyVecEnv([make_env] * 1), n_stack=4))


model_path = "model (12).zip"


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


try:
    eval_info = evaluate_policy(
        model,
        venv,
        n_eval_episodes=1,
        return_episode_rewards=True,
        deterministic=True,
        render=True,
    )

    for key, value in eval_info.items():
        print(f"{key}:  {np.mean(value)} {value}")

finally:
    venv.close()
