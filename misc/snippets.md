# Snippets

Snippets of code I forget about alot

## Integrate custom env
```python
import retro
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

retro.data.Integrations.add_custom_path(os.path.join(SCRIPT_DIR, "custom_integrations"))
env = retro.make("SimCity-Snes", inttype=retro.data.Integrations.ALL)
```

## Basic Gym run
```python
env = retro.make(
    "HotWheelsStuntTrackChallenge-gba",
    render_mode="human"
)

obs, info = env.reset(seed=42)

while True:
    act = env.action_space.sample()

    obs, reward, term, trun, info = env.step(act)
    print(reward)
    if term or trun:
        print(info, reward)
        env.close()
        break
```

## Hot load a differernt state (only tested on GBA)
```python
# create training env
env = retro.make(
    "HotWheelsStuntTrackChallenge-gba",
    state="training_state.state",
)

# do some training...
obs, info = env.reset(seed=42)

while True:
    act = env.action_space.sample()
    obs, reward, term, trun, info = env.step(act)
    print(reward)
    if term or trun:
        print(info, reward)
        env.close()
        break

# load eval state
env.unwrapped.load_state("eval_state.state")

# reset env/emulator
env.unwrapped.data.reset()
env.unwrapped.data.update_ram()

# reset state
obs, info = env.reset(seed=42)

# do some evaluating...
while True:
    act = env.action_space.sample()
    obs, reward, term, trun, info = env.step(act)
    print(reward)
    if term or trun:
        print(info, reward)
        env.close()
        break
```