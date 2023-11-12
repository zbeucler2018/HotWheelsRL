import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from gymnasium.wrappers.time_limit import TimeLimit
import retro

from .action import HotWheelsDiscretizer, StochasticFrameSkip
from .reward import PenalizeHittingWalls



class HotWheelsWrapper(gym.Wrapper):
    """
    HotWheels preprocessings

    Specifically:

    * Adds Monitor
    * Better speed calculation
    * Makes action space discrete
    * Stochastic Frame skipping: 4 by default at 25%
    * Will terminate when crashing during a trick, by default
    * Will terminate when crashing into a wall, by default
    * Gives reward of -5 when crashing into a wall, by default
    * Uses DeepMind-like wrappers, by default. Clip the reward to {+1, 0, -1} by its sign and resizes the obs to 84x84xD
    * Can also add max step limit to env.
    """

    def __init__(
        self,
        env: gym.Env,
        frame_skip: int = 4,
        terminate_on_crash: bool = True,
        terminate_on_wall_crash: bool = True,
        wall_crash_reward: int = -5,
        use_deepmind_wrapper: bool = True,
        max_episode_steps: int|None = 5_100
    ) -> None:
    
        env = FixSpeed(env)
        env = HotWheelsDiscretizer(env)

        if frame_skip > 1: # frame_skip=1 is normal env
            env = StochasticFrameSkip(env, n=frame_skip, stickprob=0.25)
        if terminate_on_crash:
            env = TerminateOnCrash(env)
        if wall_crash_reward:
            env = PenalizeHittingWalls(env, penality=wall_crash_reward)
        if terminate_on_wall_crash:
            env = TerminateOnWallCrash(env)
        if use_deepmind_wrapper:
            env = WarpFrame(env)
            env = ClipRewardEnv(env) 
        if max_episode_steps:
            # TRex_Valley: 5100 (1700*3) frames to complete 3 laps and lose to NPCs (4th)
            env = TimeLimit(env, max_episode_steps=max_episode_steps)

        # always apply a monitor as the last wrapper
        # bc sb3's evaluate_policy() won't register 
        # TerminateOnCrash and TerminateOnWallCrash 
        # bc they are wrappers and not the "true"
        # term and trunc
        env = Monitor(env)

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


    
class TerminateOnWallCrash(gym.Wrapper):
    """
    A wrapper that ends the episode if the agent has
    hit a wall. Also applies a penality.
    """

    def __init__(self, env, penality=-5):
        super().__init__(env)
        self.crash_penality = penality

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if info["hit_wall"] == 1:
            terminated = True
            truncated = True
            reward -= self.crash_penality

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
