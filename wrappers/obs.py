import gymnasium as gym
import numpy as np
from gymnasium.core import Env


class TrimmedObservation(gym.Wrapper):
    """
    Crops observation such that the speed dial,
    mini map, and lap/race timer are not included.
    Resulting obs shape is (130, 120)
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation[30:240, 55:175], reward, terminated, truncated, info


class MiniMapObservation(gym.Wrapper):
    """
    Reduces the obs to just the minimap.
    Resulting size is (65, 55, 3)
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation[94:160, 0:55], reward, terminated, truncated, info


class CropObservation(gym.Wrapper):
    """
    Reduces observation such that not
    important things (speed dial, mini map)
    are not included in the observation.
    Resulting obs shape is (110, 130, 3)
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation[50:, 50:180, :], reward, terminated, truncated, info
