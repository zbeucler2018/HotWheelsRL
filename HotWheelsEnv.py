
from dataclasses import dataclass
from enum import Enum
from typing import Any, Tuple, Union

import retro
from gymnasium.core import Env
from gymnasium.wrappers import FrameStack, GrayScaleObservation
from retro import Actions

from gym_wrappers import FixSpeed, SingleActionEnv, TerminateOnCrash, EncourageTricks, NorrmalizeBoost


class GameStates(Enum):
    SINGLE = "dino_single.state"
    SINGLE_POINTS = "dino_single_points.state"
    MULTIPLAYER = "dino_multiplayer.state"


@dataclass
class CustomEnv():
    """ Interface to make a HotWheels env """
    game_state: GameStates
    framestack: bool
    grayscale: bool
    action_space: Actions = Actions.FILTERED
    strict_action_space: bool = False
    encourage_tricks: bool = False






class HotWheelsEnvFactory():

    @staticmethod    
    def make_env(env_config: CustomEnv) -> Env:
        """
        Returns a env of a specific configuration
        """
        
        _env = retro.make(
            game="HotWheelsStuntTrackChallenge-gba", 
            render_mode="rgb_array", 
            state=env_config.game_state.value,
            use_restricted_actions=env_config.action_space
        )
        
        _env = TerminateOnCrash(_env)
        _env = FixSpeed(_env)
        # _env = NormalizeBoost(_env)

        if env_config.strict_action_space:
            _env = SingleActionEnv(_env)

        if env_config.grayscale:
            _env = GrayScaleObservation(_env, keep_dim=True)

        if env_config.framestack:
            _env = FrameStack(_env, num_stack=4)

        if env_config.encourage_tricks:
            _env = EncourageTricks(_env)


        return _env
