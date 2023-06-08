
from enum import Enum
from typing import Any, Tuple, Union

import gymnasium as gym
import numpy as np
import retro
from gymnasium.core import Env
from gymnasium.wrappers import FrameStack, GrayScaleObservation, TimeLimit


from dataclasses import dataclass



class GameStates(Enum):
    SINGLE = 'dino_single.state'
    SINGLE_POINTS = 'dino_single_points.state'
    MULTIPLAYER = 'dino_multiplayer.state'


@dataclass
class CustomEnv():
    game_state: GameStates
    discrete: bool
    multibinary: bool
    framestack: bool
    grayscale: bool
    raw: bool





class HotWheelsEnvFactory():

    @staticmethod    
    def make_env(env_config: CustomEnv) -> Env:
        """
        Returns a env of a specific configuration
        """

        if env_config.discrete and env_config.multibinary:
            raise Exception(f"HotWheelsEnv action space can only be either discrete or multibinary")
        
        _env = retro.make(
            game="HotWheelsStuntTrackChallenge-gba", 
            render_mode="rgb_array", 
            state=env_config.game_state.value
        )

        if env_config.discrete:
            _env = SingleActionEnv(_env)

        if env_config.grayscale:
            _env = GrayScaleObservation(_env, keep_dim=True)

        if env_config.framestack:
            _env = FrameStack(_env, num_stack=4)

        if not env_config.raw:
            _env = TerminateOnCrash(_env)
            _env = FixSpeed(_env)

        return _env














    
class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    Args:
        buttons: ordered list of buttons, corresponding to each dimension of the MultiBinary action space
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, buttons, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def discrete_to_boolean_array(self, action):
        """
        convert discrete action to boolean array
        """
        return self._decode_discrete_action[action].copy()
    
    def discrete_to_multibinary(self, action):
        """
        converts discrete value to multibinary array
        """
        arr = self.action(action)
        # convert boolean array to multibinary
        return arr.astype(np.uint8)


class FixSpeed(gym.Wrapper):
    """
    Fixes env bug so the speed is accurate
    """
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info['speed'] *= 0.702
        return observation, reward, terminated, truncated, info


class DoTricks(gym.Wrapper):
    """
    Encourages the agent to do tricks (increase score)
    
    """
    def __init__(self, env, score_boost=1.0, use_dynamic_reward=True):
        super().__init__(env)
        self.prev_score = None
        self.score_boost = score_boost
        self.use_dynamic_reward = use_dynamic_reward
        
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        # Get the current score and compare it with the previous score
        curr_score = info.get('score')
        if curr_score is not None and self.prev_score is not None:
            if curr_score > self.prev_score:
                if self.use_dynamic_reward:
                    reward += (1 / (curr_score - self.prev_score))
                else:
                    reward += self.score_boost
        # Update the previous score
        self.prev_score = curr_score
        return observation, reward, terminated, truncated, info


class SingleActionEnv(Discretizer):
    """
    Restricts the agent's actions to a a single button per action
            
            []
            , ['B']
            , ['A']
            , ['UP']
            , ['DOWN']
            , ['LEFT']
            , ['RIGHT']
            , ['L', 'R']
    """

    def __init__(self, env):
        super().__init__(env=env, 
                        buttons=env.unwrapped.buttons, 
                        combos=[  
            []
            , ['B']
            , ['A']
            , ['UP']
            , ['DOWN']
            , ['LEFT']
            , ['RIGHT']
            , ['L', 'R']
        ])

        self.original_env = env
    
    def get_discrete_button_meaning(self, action):
        """
        get button from discrete action
        """
        multibinary_action = self.discrete_to_multibinary(action)
        return self.original_env.get_action_meaning(multibinary_action)


class TerminateOnCrash(gym.Wrapper):
    """
    A wrapper that ends the episode if the mean of the observation is above a certain threshold
    """
    def __init__(self, env, threshold=238):
        super().__init__(env)
        self.crash_restart_threshold = threshold

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        mean_obs = observation.mean()
        if mean_obs > self.crash_restart_threshold:
            terminated = True
        return observation, reward, terminated, truncated, info
    

class NorrmalizeBoost(gym.Wrapper):
    """
    Normalizes the raw boost variable. True if boost is avaliable, false if not
    """
    def __init__(self, env):
        super().__init__(env)
        self.full_boost_quantity = 980

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info['boost'] = (True if info['boost'] == self.full_boost_quantity else False)
        return observation, reward, terminated, truncated, info



class PenalizeHittingWall(gym.Wrapper):
    """
    Penalizes the agent for hitting a wall during the episode.
    The current way to detect a wall collision is if the speed is 
    zero while progressing through the episode
    """

    def __init__(self, env, penalty):
        super().__init__(env, penalty=1)
        self.hit_wall_penality = penalty

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if info['speed'] < 5 and info['progress'] > 0:
            reward -= self.hit_wall_penality
        return observation, reward, terminated, truncated, info
    








