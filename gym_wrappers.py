import gymnasium as gym
import numpy as np
from gymnasium.core import Env
from retro import Actions
import wandb


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
        super().__init__(
            env=env,
            buttons=env.unwrapped.buttons,
            combos=[
                [],
                ["B"],
                ["A"],
                ["UP"],
                ["DOWN"],
                ["LEFT"],
                ["RIGHT"],
                ["L", "R"],
            ],
        )

        self.original_env = env

    def get_discrete_button_meaning(self, action):
        """
        get button from discrete action
        """
        multibinary_action = self.discrete_to_multibinary(action)
        return self.original_env.get_action_meaning(multibinary_action)


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
        if info["speed"] < 5 and info["progress"] > 0:
            reward -= self.hit_wall_penality
        return observation, reward, terminated, truncated, info
    

# class CalcAverageSpeed(gym.Wrapper):
#     """
#     Calculate average speed and log to wandb
#     """

#     def __init__(self, env: Env):
#         super().__init__(env)
#         self.total_speed = 0
#         self.total_steps = 0

#     def reset(self, **kwargs):
#         self.total_speed = 0
#         self.total_steps = 0
#         return self.env.reset(**kwargs)

#     def step(self, action):
#         observation, reward, terminated, truncated, info = self.env.step(action)
#         speed = info.get("speed", 0.0)
#         self.total_speed += speed
#         self.total_steps += 1
#         average_speed = self.total_speed / self.total_steps
#         info["average_speed"] = average_speed
#         return observation, reward, terminated, truncated, info
    

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
        _log = {f"Episode {self.episode_count} Info": info}
        gym.logger.info(_log)
        wandb.log(_log)
        return observation, reward, terminated, truncated, info

