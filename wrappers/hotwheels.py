import gymnasium as gym
import numpy as np
from gymnasium.core import Env


class HotWheelsWrapper(gym.Wrapper):
    """
    Allows access to RetroEnv.data
    """

    def __init__(self, env):
        super().__init__(env)

    def reset_emulator_data(self):
        """
        Resets the emulator by reseting the variables
        and updating the RAM
        """
        try:  # Bare RetroEnv
            retro_data = self.env.unwrapped.data
        except AttributeError:  # Recursively found GameData
            retro_data = self.get_wrapper_attr(name="data")
        retro_data.reset()
        retro_data.update_ram()


class StochasticFrameSkip(gym.Wrapper):
    """
    Frameskip with randomness.
    n: frames to skip
    stickprob: potential to pick another action and use it instead
    """

    def __init__(self, env: gym.Env, n, stickprob):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env.unwrapped, "supports_want_render")

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        terminated = False
        truncated = False
        totrew = 0
        for i in range(self.n):
            # First step after reset, use action
            if self.curac is None:
                self.curac = ac
            # First substep, delay with probability=stickprob
            elif i == 0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            # Second substep, new action definitely kicks in
            elif i == 1:
                self.curac = ac
            if self.supports_want_render and i < self.n - 1:
                ob, rew, terminated, truncated, info = self.env.step(
                    self.curac,
                    want_render=False,
                )
            else:
                ob, rew, terminated, truncated, info = self.env.step(self.curac)
            totrew += rew
            if terminated or truncated:
                break
        return ob, totrew, terminated, truncated, info


class FixSpeed(gym.Wrapper):
    """
    Fixes env bug so the speed is accurate
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info["speed"] *= 0.702
        return observation, reward, terminated, truncated, info


class TerminateOnCrash(gym.Wrapper):
    """
    A wrapper that ends the episode if the mean of the
    observation is above a certain threshold.
    Also applies a penality.
    Triggered when screen turns white after crash
    """

    def __init__(self, env, threshold=238, penality=-5):
        super().__init__(env)
        self.crash_restart_obs_threshold = threshold
        self.crash_penality = penality

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        mean_obs = observation.mean()
        if mean_obs >= self.crash_restart_obs_threshold:
            terminated = True
            truncated = True
            reward -= self.crash_penality

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
        info["boost"] = True if info["boost"] == self.full_boost_quantity else False
        return observation, reward, terminated, truncated, info
