import gymnasium as gym
import numpy as np
from gymnasium.core import Env


class EncourageTricks(gym.Wrapper):
    """
    Encourages the agent to do tricks (increase score)
    """

    def __init__(self, env):
        super().__init__(env)
        self.prev_score = None

    def reset(self, **kwargs):
        self.prev_score = None
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        # Get the current score and compare it with the previous score
        curr_score = info.get("score")
        if curr_score is not None and self.prev_score is not None:
            if curr_score > self.prev_score:
                reward += 1 / (curr_score - self.prev_score)
        # Update the previous score
        self.prev_score = curr_score
        return observation, reward, terminated, truncated, info


class IncreaseMeanSpeed(gym.Wrapper):
    """
    Gives reward if mean speed has been increased
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._env = env
        self.speeds = []
        self.mean_speed = 0

    def reset(self, **kwargs):
        self.speeds = []
        self.mean_speed = 0
        return super().reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self._env.step(action)

        self.speeds.append(info["speed"])

        # calc new mean speed
        new_mean_speed = np.mean(self.speeds)

        # compare to old mean speed
        if new_mean_speed > self.mean_speed:
            reward += 0.1
        else:
            reward -= 0.1

        # update mean speed
        self.mean_speed = new_mean_speed

        return observation, reward, terminated, truncated, info


class PenalizeHittingWalls(gym.Wrapper):
    """
    Penalizes the agent for
    hitting a wall
    """

    def __init__(self, env, penality: int = -5):
        self.hit_wall_penality = penality
        super().__init__(env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        # TODO: make this better to work on all tracks
        if info["hit_wall"] == 1:
            reward -= self.hit_wall_penality
        return observation, reward, terminated, truncated, info
