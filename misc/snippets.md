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