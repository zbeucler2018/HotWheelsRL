import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from gymnasium.wrappers.time_limit import TimeLimit

from .action import HotWheelsDiscretizer, StochasticFrameSkip
from .reward import PenalizeHittingWalls


class HotWheelsWrapper(gym.Wrapper):
    """
    Allows access to RetroEnv.data
    """

    def __init__(
        self,
        env: gym.Env,
        frame_skip: int = 4,
        terminate_on_crash: bool = True,
        terminate_on_wall_crash: bool = True,
        wall_crash_reward: int = -5,
        use_deepmind_wrapper: bool = True,
        max_episode_steps: int|None = None
    ) -> None:
    
        env = Monitor(env)
        env = FixSpeed(env)
        env = HotWheelsDiscretizer(env)

        if frame_skip > 1: # frame_skip=1 is normal env
            env = StochasticFrameSkip(n=frame_skip, stickprob=0.25)
        if terminate_on_crash:
            env = TerminateOnCrash(env)
        if wall_crash_reward:
            env = PenalizeHittingWalls(env, penality=wall_crash_reward)
        # if terminate_on_wall_crash:
        #     env = TerminateOnWallCrash(env)
        if use_deepmind_wrapper:
            env = WarpFrame(env)     # Resize obs to 84x84xD
            env = ClipRewardEnv(env) # Clip the reward to {+1, 0, -1} by its sign
        if max_episode_steps:
            # TRex_Valley: 5100 (1700*3) frames to complete 3 laps and lose to NPCs (4th)
            env = TimeLimit(env, max_episode_steps=max_episode_steps)

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
