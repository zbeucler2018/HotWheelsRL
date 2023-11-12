import retro
from gymnasium.wrappers import (
    GrayScaleObservation,
    ResizeObservation,
    NormalizeReward,
    FrameStack,
    NormalizeObservation,
)
from callbacks.eval_policy import evaluate_policy
# from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
    DummyVecEnv,
)
from wrappers.viewer import Viewer
import numpy as np
from utils import HotWheelsStates
import train_utils
from wrappers.hotwheels import HotWheelsWrapper


def make_env():
    _game = "HotWheelsStuntTrackChallenge-gba"
    _state = HotWheelsStates.DINO_BONEYARD_MULTI
    _rm = "human"
    env = train_utils.make_retro(
        game=_game, state=_state, scenario=None, render_mode=_rm
    )
    env = HotWheelsWrapper(env)  # allows us to change to eval state
    #     env = Viewer(env)
    return env


venv = VecTransposeImage(VecFrameStack(DummyVecEnv([make_env] * 1), n_stack=4))


model_path = "model (11).zip"


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
        n_eval_episodes=2,
        return_episode_rewards=True,
        deterministic=False,
        render=False,
    )

    for key, value in eval_info.items():
        print(f"{key}:  {np.mean(value)}")

finally:
    venv.close()
