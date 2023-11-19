from utils import HotWheelsStates
import retro
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from wrappers.hotwheels import StochasticFrameSkip, TerminateOnCrash
from wrappers.reward import PenalizeHittingWalls
from wrappers.action import HotWheelsDiscretizer


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
    return env


def wrap_deepmind_retro(env):
    """
    Configure environment for retro games, using config similar to DeepMind-style Atari in openai/baseline's wrap_deepmind
    """
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    return env
