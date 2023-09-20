import retro
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import (
    GrayScaleObservation,
    ResizeObservation,
    NormalizeReward,
    FrameStack,
    NormalizeObservation,
)
from gym_wrappers import *
from callbacks.eval_policy import evaluate_policy

# from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
    DummyVecEnv,
)
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame

from viewer import Viewer

import numpy as np


from utils import HotWheelsStates


def make_retro(
    *,
    game,
    state: HotWheelsStates = HotWheelsStates.DEFAULT,
    max_episode_steps=4500,
    render_mode="rgb_array",
    **kwargs,
):
    print(f"Using ", f"state: {state}.state", f"info: {state}.json", sep="\n")
    env = retro.make(
        game,
        state=f"{state}.state",
        info=retro.data.get_file_path(
            "HotWheelsStuntTrackChallenge-gba", f"{state}.json"
        ),
        render_mode=render_mode,
        **kwargs,
    )
    env = Monitor(env)
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    env = TerminateOnCrash(env)
    env = PenalizeHittingWalls(env)
    env = HotWheelsDiscretizer(env)

    return env


def wrap_deepmind_retro(env):
    """
    Configure environment for retro games, using config similar to DeepMind-style Atari in openai/baseline's wrap_deepmind
    """
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    return env


def make_env():
    env = make_retro(
        game="HotWheelsStuntTrackChallenge-gba",
        state="TRex_Valley_single",
        scenario=None,
    )
    env = wrap_deepmind_retro(env)
    env = Viewer(env)
    return env


venv = VecTransposeImage(VecFrameStack(DummyVecEnv([make_env] * 1), n_stack=4))


model_path = "best_model (3).zip"


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
        render=False,
    )

    for key,value in eval_info.items():
        print(f"{key}:  {np.mean(value)}")

finally:
    venv.close()
