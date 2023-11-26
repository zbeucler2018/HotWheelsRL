import gymnasium as gym
import numpy as np
from gymnasium.core import Env


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
            [],
            ["A", "UP"],
            ["A", "DOWN"],
            ["A", "LEFT"],
            ["A", "RIGHT"],
            ["A", "L", "R"],
    """

    def __init__(
        self,
        env,
        action_space=[
            [],
            ["A", "UP"],
            ["A", "DOWN"],
            ["A", "LEFT"],
            ["A", "RIGHT"],
            ["A", "L", "R"],
        ],
    ):
        super().__init__(env=env, combos=action_space)
