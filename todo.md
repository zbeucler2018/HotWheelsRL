# Todos
- [ x eval should use start state
    - [x] seamlessly switch from training state to eval state and back
    - [x] pass training states via cli
        - RN it just uses 232.state for all envs
    - [x] pass eval state via cli
- [ ] create 8 different states on the track
- [x] modify training vec env creation st it uses a different state per env
    - eval freq cli arg (100k default || max(args.num_steps // args.num_envs, 1) )
- [ ] adapt new project structure
```
# where do I put HotWheelsStates so everything has access to it?
|- wrappers/
    |- __init__.py
    |- obs.py       (CropObservation, MiniMapObservation, NavObservation)
    |- action.py    (Discretizer, HotWheelsDiscretizer, StochasticFrameSkip)
    |- reward.py    (EncourageTricks, IncreaseMeanSpeed, PenalizeHittingWalls)
    |- hotwheels.py (HotWheelsWrapper, NormalizeBoost, TerminateOnCrash, TerminateOnWallCrash, FixSpeed)
|- utils/
    |- __init__.py
    |- config.py       (arg parser and json/yaml support)
    |- misc.py
    |- training.py     (???)
    |- interactive.py
    |- viewer.py
|- evaluation/
    |- __init__.py
    |- evalCallback.py
    |- evalPolicy.py
    |- utils.py        (evaluate_policy_on_state())
|- HotWheelsStuntTrackChallenge-gba/
|- train.py
|- notebook.ipynb
|- test.py
|- docs/ (benchmarks, snippets, notes, images/gifs)
|- readme.md
```


```python
import gymnasium as gym
import retro

def make_hotwheels(): # ?
    _env = make_retro(...)
    return HotWheelsWrapper(_env, ...)


def make_retro(state: HotWheelsStates, render_mode, **kwargs) -> gym.Env:
    """Makes a basic hot wheels retro env"""
    return retro.make(
        game,
        state=f"{state}.state",
        info=retro.data.get_file_path(
            "HotWheelsStuntTrackChallenge-gba", f"{state}.json"
        ),
        render_mode=render_mode,
        **kwargs,
    )


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
        if terminate_on_wall_crash:
            env = TerminateOnWallCrash(env)
        if use_deepmind_wrapper:
            env = WarpFrame(env)     # Resize obs to 84x84xD
            env = ClipRewardEnv(env) # Clip the reward to {+1, 0, -1} by its sign
        if max_episode_steps:
            # TRex_Valley: 5100 (1700*3) frames to complete 3 laps and lose to NPCs (4th)
            env = TimeLimit(env, max_episode_steps=max_episode_steps)

        super().__init__(env)


    def reset_emulator_data(self) -> None:
        """
        Resets the emulator by reseting the variables
        and updating the RAM
        """
        # maybe add load_state() too but idk
        retro_data = self.get_wrapper_attr(name="data")
        retro_data.reset()
        retro_data.update_ram()
```


```python
from dataclasses import dataclass

@dataclass
class Config:
    # model
    total_steps: int
    num_envs: int
    resume: bool = False
    run_id: any = None
    model_path: str

    # env
    game: str = "HotWheelsStuntTrackChallenge-gba"
    state: HotWheelsStates = HotWheelsStates.DEFAULT
    scenario: any = None
    # what else?
```
