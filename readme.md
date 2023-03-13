# HotWheelsRL (wip)

Zack Beucler


- Use RL to train an agent to competitively complete a single lap on the first level in the GBA game 'Hot Wheels Stunt Track Challenge'
- Agent should be decently fast, and increase it's score

## Dev env
- run `start.bash` and open the url in browser. Use `start.bash --build` for fresh install
  - make sure to run `!bash import_rom_into_retro.bash` before using `import retro`

## Todo
- benchmarks
  - [ ] random agent
  - [ ] NPCs (of different difficulties if possible)
  - [ ] human
  - [ ] (bonus) make static site for leaderboards 
      - record video of each agent
      - save model
      - save time
      - use github pages to host it
- wrappers
  - [ ] removes 'SELECT' and 'START' buttons from action space
  - [ ] encourages less frames / timesteps
  - [ ] encourages tricks
  - [ ] penalizes bumping into the wall repeatdly
- dev env
  - [ ] import rom into retro in dockerfile if possible



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
  - [ ] restart cutscene
  - [x] speed
  - [ ] multiplayer rank
  - [ ] figure out a way to port the integration UI to ubuntu 22
    - libretro or QT is out of date (?)
