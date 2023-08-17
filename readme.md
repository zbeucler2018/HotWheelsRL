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

```math
\sum_{i=1}^{n} (d_progress * 0.8) + (d_speed * 0.2)
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

## Integration

### dino1_single

  - Time_s:
    - game time on screen updates in increments of 5ms, every 3 frames
    - (`300203c` , `=u4`) increments 2 every 3 frames
    - get time by getting total ms
    - `dino_single.state` starts at 25ms already
      - 140 or 142

### dino_multi

  - starts at 21ms already
  - progress
    - 2007100, =i2
  - rank
    - 2008d60 (33590624), |i1
    - retro has a hard time with this one I think. Worst case, extract with lua
    - 0 is 1st, 3 is 4th
  - boost
    - 20070a8, (33583272), <=i4
    - full at 980, empty at 0
  - score
    - my_score: 2007208, <=i4
    - npc_score: 2007c16, >=u4 (seems to be all of theirs??)
  - lap
    - 20070f2, |u1
  - speed
    - 20070a1, <u2
    - **still off like in dino1_single**
  - hit wall
    - 20070f0, ><n4,
    - related to lap and rank vars
    - (rank * 1000) + lap = False, ((rank * 1100) + lap) = True
    - if in fourth place and on second lap and collided with wall, than ram value is 3102
      - hit_wall = data.hit_wall == (data.rank * 1100) + lap

### dino2_multi
  - progress
  - hit_wall
    - 100 = True if in 1st place
  - lap
  - score
  - boost
  - npc_score
    - same as dino1_multi
  - rank
    - can be extracted from hit_wall
    - str(hit_wall)[-1]
  - laps
    - extracted from progress
      - 1 lap 344 (probably 345 is better but it s little over the line)  
