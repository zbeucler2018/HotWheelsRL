
from enum import Enum
from typing import Tuple, Union, Any

import numpy as np

import gymnasium as gym
from gymnasium.core import Env
from gymnasium.wrappers import GrayScaleObservation, TimeLimit, FrameStack

import retro


class GameStates(Enum):
    SINGLE = 'dino_single.state',
    SINGLE_POINTS = 'dino_single_points.state',
    MULTIPLAYER = 'dino_multiplayer.state'

class Actions(Enum):
    MULTI = 0,
    DISCRETE = 1

class CustomEnvs(Enum):
    BARE_DISCRETE = -2,
    BARE_MULTI = -1, # no wrappers or anything

    MULTI = 1,
    MULTI_FRAMESTACK = 2
    MULTI_GRAY = 3,
    MULTI_GRAY_FRAMESTACK = 4,

    DISCRETE = 5,
    DISCRETE_FRAMESTACK = 6,
    DISCRETE_GRAY = 7,
    DISCRETE_GRAY_FRAMESTACK = 8



class HotWheelsEnv():

    def __init__(self, multiplayer=True):
        self.is_multiplayer = multiplayer
        self.game_state: GameStates = GameStates.MULTIPLAYER if self.is_multiplayer else GameStates.SINGLE 
        self.max_episode_steps = 25_000

    def make_base_env(self) -> Env:
        """
        Returns a basic, protected env
        """
        env = retro.make(game="HotWheelsStuntTrackChallenge-gba", render_mode="rgb_array", state=self.game_state.value)
        env = TimeLimit(env, max_episode_steps=self.max_episode_steps)
        env = TerminateOnCrash(env)
        env = FixSpeed(env)
        #env = NorrmalizeBoost(env)
        return env

    def make_custom_env(self, custom_env: CustomEnvs) -> Env:
        """
        Returns a custom Hot Wheels environment based on the specified wrapper configuration.

        Args:
            custom_env: The wrapper configuration to apply.

        Returns:
            The configured environment.
        """
        if custom_env in (CustomEnvs.BARE_DISCRETE, CustomEnvs.BARE_MULTI):
            env = retro.make(game="HotWheelsStuntTrackChallenge-gba", render_mode="rgb_array", state=self.game_state.value)
        else:
            env = self.make_base_env()

        if custom_env in (CustomEnvs.BARE_MULTI, CustomEnvs.MULTI):
            return env
        elif custom_env == CustomEnvs.MULTI_FRAMESTACK:
            env = FrameStack(env, num_stack=4)
        elif custom_env == CustomEnvs.MULTI_GRAY:
            env = GrayScaleObservation(env, keep_dim=True)
        elif custom_env == CustomEnvs.MULTI_GRAY_FRAMESTACK:
            env = GrayScaleObservation(env, keep_dim=True)
            env = FrameStack(env, num_stack=4)
        elif custom_env in (CustomEnvs.DISCRETE, CustomEnvs.DISCRETE_FRAMESTACK, CustomEnvs.DISCRETE_GRAY, CustomEnvs.DISCRETE_GRAY_FRAMESTACK):
            env = SingleActionEnv(env)
            if custom_env in (CustomEnvs.DISCRETE_FRAMESTACK, CustomEnvs.DISCRETE_GRAY_FRAMESTACK):
                env = FrameStack(env, num_stack=4)
            if custom_env in (CustomEnvs.DISCRETE_GRAY, CustomEnvs.DISCRETE_GRAY_FRAMESTACK):
                env = GrayScaleObservation(env, keep_dim=True)
        else:
            raise NotImplementedError("How did you get here?")
        
        return env














    
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



"""
monitor wrapper : speed, progress, score, rank, etc
"""
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
    








