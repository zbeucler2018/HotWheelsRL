# HotWheelsRL (wip)

Zack Beucler


- Use RL to train an agent to competitively complete a single lap on the first level in the GBA game 'Hot Wheels Stunt Track Challenge'
- Agent should be decently fast, and increase it's score

## Dev env
- run `start.bash` and open the url in browser. Use `start.bash --build` for fresh install
  - make sure to run `!bash import_rom_into_retro.bash` before using `import retro`

## Todo
- benchmarks
  - [ ] algorithms
    - [ ] [A2C](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html)
      - [ ] mlp
      - [ ] cnn
    - [ ] [PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
      - [ ] mlp
      - [ ] cnn
    - [ ] [TRPO](https://sb3-contrib.readthedocs.io/en/master/modules/trpo.html)
    - [ ] [Exploration Strategies for Deep Reinforcement Learning](https://github.com/pkumusic/E-DRL)
    - [ ] NEAT
  - [ ] random agent
  - [ ] NPCs (of different difficulties if possible)
  - [ ] human

- wrappers
  - [x] removes 'SELECT' and 'START' buttons from action space
  - [ ] ~~encourages less frames / timesteps~~ encourage higher speed
  - [x] throw terminated on restart (screen goes white)
  - [x] encourages tricks
  - [ ] penalizes bumping into the wall repeatdly
  - [ ] wrapper that normalizes boost
    - vertify if semi boost can be activated
- dev env
  - [ ] import rom into retro in dockerfile if possible
- bonus
  - [ ] leaderboard site
    - leaderboard for fastest lap amoung all agents of different algorithms
      - has a video of the race for each entry and some info about the agent
    - mkdocs + github pages
     



## Resources

- [stable_baselines](https://github.com/Stable-Baselines-Team/stable-baselines)
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
- [stable-baselines3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)
- [Exploration Strategies for Deep Reinforcement Learning](https://github.com/pkumusic/E-DRL)


### Integration
- need ubuntu 18
- variables I want from RAM
  - [x] track progress
  - [x] score
  - [x] restart cutscene
  - [x] speed
  - [x] boost
  - [ ] multiplayer rank
  - [ ] figure out a way to port the integration UI to ubuntu 22
    - libretro or QT is out of date (?)
