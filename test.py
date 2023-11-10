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


# def make_retro(
#     *,
#     game,
#     state: HotWheelsStates = HotWheelsStates.DEFAULT,
#     max_episode_steps=4500,
#     render_mode="human",
#     **kwargs,
# ):
#     print(f"Using ", f"state: {state}.state", f"info: {state}.json", sep="\n")
#     env = retro.make(
#         game,
#         state=f"{state}.state",
#         info=retro.data.get_file_path(
#             "HotWheelsStuntTrackChallenge-gba", f"{state}.json"
#         ),
#         render_mode=render_mode,
#         **kwargs,
#     )
#     env = Monitor(env)
#     env = StochasticFrameSkip(env, n=4, stickprob=0.25)
#     env = TerminateOnCrash(env)
#     env = PenalizeHittingWalls(env)
#     env = HotWheelsDiscretizer(env)

#     return env


# def wrap_deepmind_retro(env):
#     """
#     Configure environment for retro games, using config similar to DeepMind-style Atari in openai/baseline's wrap_deepmind
#     """
#     env = WarpFrame(env)
#     env = ClipRewardEnv(env)
#     return env


# def make_env():
#     env = make_retro(
#         game="HotWheelsStuntTrackChallenge-gba",
#         state=HotWheelsStates.DINO_BONEYARD_MULTI,
#         scenario=None,
#     )
#     env = wrap_deepmind_retro(env)
#     env = Viewer(env)
#     return env


# venv = VecTransposeImage(VecFrameStack(DummyVecEnv([make_env] * 1), n_stack=4))


# model_path = "model (10).zip"


# model = PPO.load(
#     path=model_path,
#     env=venv,
#     # Needed because sometimes sb3 cant find the
#     # obs and action space. Seen in colab on 8/21/23
#     custom_objects={
#         "observation_space": venv.observation_space,
#         "action_space": venv.action_space,
#     },
# )


# try:
#     eval_info = evaluate_policy(
#         model,
#         venv,
#         n_eval_episodes=1,
#         return_episode_rewards=True,
#         deterministic=True,
#         render=False,
#     )

#     for key,value in eval_info.items():
#         print(f"{key}:  {np.mean(value)}")

# finally:
#     venv.close()

# import time

# # create eval env
# state = HotWheelsStates.TREX_VALLEY_SINGLE
# env = retro.make(
#     "HotWheelsStuntTrackChallenge-gba",
#     render_mode="human",
#     state=f"{state}.state",
#     info=retro.data.get_file_path(
#         "HotWheelsStuntTrackChallenge-gba", f"{state}.json"
#     ),
# )

# full_track_state = env.unwrapped.em.get_state()

# env.close()
# del env

# # create training env stae
# env = retro.make(
#     "HotWheelsStuntTrackChallenge-gba",
#     render_mode="human",
#     state=f"{232}.state",
#     info=retro.data.get_file_path(
#         "HotWheelsStuntTrackChallenge-gba", f"{state}.json"
#     ),
# )

# obs, info = env.reset(seed=42)

# n_step = 0
# while True:
#     n_step += 1
#     print(n_step, "232", info)

#     if n_step >= 1000:
#         break

#     act = env.action_space.sample()

#     obs, reward, term, trun, info = env.step(act)
#     if term or trun:
#         env.close()
#         break

# print("loading new state")

# #env.em.set_state(full_track_state)
# env.unwrapped.load_state(f"{HotWheelsStates.TREX_VALLEY_SINGLE}.state")
# env.unwrapped.data.reset()
# env.unwrapped.data.update_ram()

# obs, info = env.reset(seed=42)


# n_step = 0
# while True:
#     n_step += 1
#     print(n_step, "full", info)

#     if n_step >= 1000:
#         break

#     act = env.action_space.sample()

#     obs, reward, term, trun, info = env.step(act)
#     if term or trun:
#         print(info, reward)
#         env.close()
#         break

#     if n_step == 1:
#         time.sleep(5)


"""



"""
import train_utils

import retro


args = {
    "game": "HotWheelsStuntTrackChallenge-gba",
    "state": "TRex_Valley_single",
    "scenario": None,
    "num_envs": 3,
    "train_states": ["TRex_Valley_single", "87", "61"],
}


def make_env():
    env = train_utils.make_retro(
        game=args["game"],
        state=args["state"],
        scenario=args["scenario"],
        render_mode="rgb_array",
    )
    env = train_utils.wrap_deepmind_retro(env)
    env = HotWheelsWrapper(env)  # allows us to change to eval state
    return env


def main():
    venv = VecTransposeImage(
        VecFrameStack(SubprocVecEnv([make_env] * args["num_envs"]), n_stack=4)
    )

    # Need to change state AFTER adding SubProcVec because
    # retro will throw "1 Emulator process only: exception
    # if applied before
    for indx, t_state in enumerate(args["train_states"]):
        _ = venv.env_method(
            method_name="load_state", statename=f"{t_state}.state", indices=indx
        )
        _ = venv.env_method(method_name="reset_emulator_data", indices=indx)
    observations = venv.reset()

    # [[HotWheelsState] ->


# def change_state(venv, statenames: [str]|[HotWheelsStates], indicies = None):
#     for indx,t_state in enumerate(statenames):
#         _ = venv.env_method(method_name="load_state", statename=f"{t_state}.state", indices=indx)
#         _ = venv.env_method(method_name="reset_emulator_data", indices=indx)


if __name__ == "__main__":
    main()
