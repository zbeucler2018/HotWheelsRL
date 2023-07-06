import unittest

import retro

from HotWheelsEnv import (
    GameStates,
    FixSpeed,
    EncourageTricks,
    TerminateOnCrash,
    NorrmalizeBoost,
    SingleActionEnv,
    LogInfoValues,
    CalcAverageSpeed
)


class TestGameStates(unittest.TestCase):
    """Tests if a env can be created with each GameState"""

    def tearDown(self):
        self.env.close()
        self.env = None

    def test_dino_single(self):
        self.env = retro.make(
            game="HotWheelsStuntTrackChallenge-gba",
            render_mode="rgb_array",
            state=GameStates.SINGLE.value,
        )

    def test_dino_single_points(self):
        self.env = retro.make(
            game="HotWheelsStuntTrackChallenge-gba",
            render_mode="rgb_array",
            state=GameStates.SINGLE_POINTS.value,
        )

    @unittest.skip(f"Skip until retro can take different data.json filenames")
    def test_dino_multi(self):
        self.env = retro.make(
            game="HotWheelsStuntTrackChallenge-gba",
            render_mode="rgb_array",
            state=GameStates.MULTIPLAYER.value,
        )


from retro import Actions


class TestWrappers(unittest.TestCase):
    def setUp(self):
        self.env = retro.make(
            game="HotWheelsStuntTrackChallenge-gba",
            render_mode="rgb_array",
            state=GameStates.SINGLE.value,
        )
        _, _ = self.env.reset(seed=42)

    def tearDown(self):
        self.env.close()
        self.env = None

    def test_FixSpeed(self):
        self.env = FixSpeed(self.env)
        random_action = self.env.action_space.sample()
        observation, reward, terminated, truncated, info = self.env.step(random_action)

    def test_DoTricks(self):
        self.env = EncourageTricks(self.env)
        random_action = self.env.action_space.sample()
        observation, reward, terminated, truncated, info = self.env.step(random_action)

    def test_TerminateOnCrash(self):
        self.env = TerminateOnCrash(self.env)
        random_action = self.env.action_space.sample()
        observation, reward, terminated, truncated, info = self.env.step(random_action)

    @unittest.skip("the change to data.json isnt working (doesnt detect boost entry)")
    def test_NorrmalizeBoost(self):
        self.env = NorrmalizeBoost(self.env)
        random_action = self.env.action_space.sample()
        observation, reward, terminated, truncated, info = self.env.step(random_action)

    @unittest.skip("broken")
    def test_SingleActionEnv(self):
        self.env = SingleActionEnv(self.env)
        random_action = self.env.action_space.sample()
        observation, reward, terminated, truncated, info = self.env.step(random_action)

    def test_LogInfoValues(self):
        self.env = LogInfoValues(self.env)
        for _ in range(100):
            random_action = self.env.action_space.sample()
            observation, reward, terminated, truncated, info = self.env.step(random_action)

    def test_CalcAverageSpeed(self):
        self.env = CalcAverageSpeed(self.env)
        for _ in range(100):
            random_action = self.env.action_space.sample()
            observation, reward, terminated, truncated, info = self.env.step(random_action)


if __name__ == "__main__":
    unittest.main()
