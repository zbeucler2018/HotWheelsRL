# HotWheelsRL (wip)

Zack Beucler


- Use RL to train an agent to competitively complete a race on the first level in the GBA game 'Hot Wheels Stunt Track Challenge'
- Agent should be decently fast (less than 2min per lap), and increase it's score

## Todo
- [ ] Train 3 agents for each algo
  - try different policies (cnn, mlp) especially to give the model speed, score, etc during training
- [ ] Re-evaluate to improve
- bonus
  - [ ] leaderboard site
    - leaderboard for fastest lap amoung all agents of different algorithms
      - has a video of the race for each entry and some info about the agent
    - mkdocs + github pages
     
**MUST HAVE A `data.json` FILE**


## Immediate TODO
  - [ ] Figure out loging info to wandb in vec envs
  - [ ] Trim observation such that only important stuff is included (wrapper)
  - [ ] After training, record video of trained agent and save as wandb artifact
  - [ ] For training, dont stop at 1 lap, allow full race (3 laps)


## Resources

- [stable_baselines](https://github.com/Stable-Baselines-Team/stable-baselines)
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
- [stable-baselines3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)
- [Exploration Strategies for Deep Reinforcement Learning](https://github.com/pkumusic/E-DRL)
- [stable-retro](https://github.com/Farama-Foundation/stable-retro)



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
\sum_{i=1}^{n} (progress_i - progress_{i-1}) + (1 / (score_i - score_{i-1}))
```

  - `n` : Total time steps in episode
  - In my mind, this should encourage the bot to make forward progress and score points. I had to normalize the score reward since its usually around ~1000
    - should figure out how to encourage the bot to complete laps faster


#### Experimental reward function
- train 3 laps
- `+10` for completing a lap
- `+0.1` or `+0.01` for increasing speed
- bigger score reward