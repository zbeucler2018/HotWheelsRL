from evaluation.eval_policy import evaluate_policy

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

import os

def delete_files_containing_string(search_string):
    current_directory = os.getcwd()

    for filename in os.listdir(current_directory):
        if search_string in filename:
            file_path = os.path.join(current_directory, filename)
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")


TRIALS = 100
sucesses = 0

FAST_AC = [
    []
    ,["A", "UP"]
    ,["A", "DOWN"]
    ,["A", "LEFT"]
    ,["A", "RIGHT"]
    ,["A", "L", "R"]
]
SLOW_AC = [
    []
    ,["A"]
    ,["UP"]
    ,["DOWN"]
    ,["LEFT"]
    ,["RIGHT"]
    ,["L", "R"]
]

def make_env():
    env = make_retro(
        game="HotWheelsStuntTrackChallenge-gba",
        state=HotWheelsStates.TREX_VALLEY_MULTI,
        render_mode="rgb_array"
    )
    env = HotWheelsWrapper(
        env,
        action_space=FAST_AC,
        terminate_on_wall_crash=True,
        terminate_on_crash=True
    )
    #env = Viewer(env)
    return env

def main(i):

    global sucesses

    venv = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_env] * 1), n_stack=4))
    # venv = VecVideoRecorder(venv, "./", lambda x: x == 0, video_length=5100, name_prefix=f"{i}-dbm_slow_20m")

    model_path = "tvm_fast_test.zip"


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

        print(f"---- {i} ----")
        for key, value in eval_info.items():
            print(f"{key}:  {np.mean(value)} {value} ")
            
        

    finally:
        venv.close()


    if eval_info['episode_laps'] == [11]:
        print(f"{i} worked!")
        sucesses = sucesses + 1
if __name__ == "__main__":
    for i in range(TRIALS):
        main(i)
        print(f"{i} *** won {sucesses}/{TRIALS}")
    print(sucesses)