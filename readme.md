# HotWheelsRL (wip)

<img src="misc/15-dbm_slow_20m_flawless.gif" style="display: block; margin-left: auto; margin-right: auto;">

Zack Beucler

- Use RL to train an agent to competitively play the first 2 tracks in the GBA game 'Hot Wheels Stunt Track Challenge'

## Resources

- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
- [Exploration Strategies for Deep Reinforcement Learning](https://github.com/pkumusic/E-DRL)
- [stable-retro](https://github.com/Farama-Foundation/stable-retro)


# Reward function

$$
R'(progress, didHitWall, didCrash) = \min\left(\max\left(\begin{cases}
+1 & \text{if } \Delta progress > 0 \\
-5 & \text{if } didHitWall \\
-5 & \text{if } didCrash \\
\end{cases}, -1\right), 1\right)
$$




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