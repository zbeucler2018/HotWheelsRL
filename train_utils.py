from utils import HotWheelsStates
import retro
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from gym_wrappers import *


def make_retro(
    *,
    game,
    state: HotWheelsStates = HotWheelsStates.DEFAULT,
    max_episode_steps=5100,
    render_mode="rgb_array",
    **kwargs,
):
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

    if max_episode_steps is not None:
        # 5100 (1700*3) frames to complete 3 laps (trex valley) and be 4th vs NPCs
        env = TimeLimit(env, max_episode_steps=max_episode_steps)

    print(f"Using ", f"state: {state}.state", f"info: {state}.json", sep="\n")
    return env


def wrap_deepmind_retro(env):
    """
    Configure environment for retro games, using config similar to DeepMind-style Atari in openai/baseline's wrap_deepmind
    """
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    return env
