import gymnasium as gym
import numpy as np
from gymnasium.core import Env
from retro import Actions
import wandb


class Discretizer(gym.ActionWrapper):
    def __init__(self, env: Env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, _action):
        return self._decode_discrete_action[_action].copy()


class HotWheelsDiscretizer(Discretizer):
    """
    Forces Agent to use specific buttons and combos
    """

    def __init__(self, env):
        action_space = [
            ["A", "UP"],
            ["A", "DOWN"],
            ["A", "LEFT"],
            ["A", "RIGHT"],
            ["A", "L", "R"],
        ]
        super().__init__(env=env, combos=action_space)


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


class TerminateOnCrash(gym.Wrapper):
    """
    A wrapper that ends the episode if the mean of the observation is above a certain threshold.
    FYI: The screen will turn completely white when restarting.
    """

    def __init__(self, env, threshold=238):
        super().__init__(env)
        self.crash_restart_threshold = threshold

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        mean_obs = observation.mean()
        if mean_obs >= self.crash_restart_threshold:
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
        info["boost"] = True if info["boost"] == self.full_boost_quantity else False
        return observation, reward, terminated, truncated, info


class LogInfoValues(gym.Wrapper):
    """
    logs all the values from the info dict to wandb
    """

    def __init__(self, env: Env):
        super().__init__(env)
        self.episode_count = 0

    def reset(self, **kwargs):
        self.episode_count += 1
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        gym.logger.info(info)
        # wandb.log(info)
        return observation, reward, terminated, truncated, info


class PunishHittingWalls(gym.Wrapper):
    """
    Punishes agent when hitting a wall
    """

    def __init__(self, env: Env, punishment: int = 5):
        super().__init__(env)
        self.punishment = punishment

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if info.get("hit_wall", None) and info.get("hit_wall", None) == 101:
            reward -= self.punishment

        return observation, reward, terminated, truncated, info


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
