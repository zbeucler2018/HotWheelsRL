# HotWheelsRL (wip)

<img src="misc/vec.gif" style="display: block; margin-left: auto; margin-right: auto;">

Zack Beucler

- Use RL to train an agent to competitively complete a race on the first level in the GBA game 'Hot Wheels Stunt Track Challenge'
- Agent should be be able to complete a lap

## Resources

- [stable_baselines](https://github.com/Stable-Baselines-Team/stable-baselines)
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
- [stable-baselines3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)
- [Exploration Strategies for Deep Reinforcement Learning](https://github.com/pkumusic/E-DRL)
- [stable-retro](https://github.com/Farama-Foundation/stable-retro)

- NOTE: `actual_playing.state` is a save file of other maps and challenges in the game.

|              | discrete | multidiscrete | multibinary | 
| ------------ | -------- | ------------- | ----------- |
| PPO          | ✅       | ✅            | ✅          |
| A2C          | ✅       | ✅            | ✅          |
| DQN          | ✅       | ❌            | ❌          |
| HER          | ✅       | ❌            | ❌          |
| QR-DQN       | ✅       | ❌            | ❌          |
| RecurrentPPO | ✅       | ✅            | ✅          |
| TRPO         | ✅       | ✅            | ✅          |
| Maskable PPO | ✅       | ✅            | ✅          |
| ARS          | ✅       | ❌            | ❌          |

# Reward function
- math is probably formatted wrong but idc
- speed reward:
  - +/- 0.1 if mean speed increases/decreases

```math
\sum_{i=1}^{n} \delta progress
```

  - `n` : Total time steps in episode
  - In my mind, this should encourage the bot to make forward progress and score points

#### Experimental reward function
- train 3 laps
- `+10` for completing a lap
- `+0.1` or `+0.01` for increasing speed
- bigger score reward

## Hyperparameters
- Using PPO hyperparameters from [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf) paper
```
learning_rate=2.5e-4,
n_steps=128,
n_epochs=3,
batch_size=32,
ent_coef=0.01,
vf_coef=1.0,
num_envs=8
```