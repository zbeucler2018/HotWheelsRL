import gymnasium as gym
import numpy as np


class Discretizer(gym.ActionWrapper):
    def __init__(self, env: gym.Env, combos):
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

    def __init__(self, env):
        action_space = [
            [],
            ["A", "UP"],
            ["A", "DOWN"],
            ["A", "LEFT"],
            ["A", "RIGHT"],
            ["A", "L", "R"],
        ]
        super().__init__(env=env, combos=action_space)


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
