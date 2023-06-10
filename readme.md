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

