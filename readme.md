# HotWheelsRL (wip)

Zack Beucler


- Use RL to train an agent to competitively complete a single lap on the first level in the GBA game 'Hot Wheels Stunt Track Challenge'

- [stable_baselines](https://github.com/Stable-Baselines-Team/stable-baselines)
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
- [stable-baselines3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)
- [Exploration Strategies for Deep Reinforcement Learning](https://github.com/pkumusic/E-DRL)


- my local dev pip env is requirements.txt


# install retro on ubuntu 22


## install python3.7 (other versions might work)
```bash
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.7
```
## install python3.7 venv
```bash
sudo apt-get install python3.7-venv
```
## make python3.7 venv
```bash
python3.7 -m venv env
source env/bin/activate
```
## install retro
```bash
pip install gym-retro
```
- `pip list` output
```
Package            Version
------------------ -------
cloudpickle        2.2.1
gym                0.25.2
gym-notices        0.0.8
gym-retro          0.8.0
importlib-metadata 6.0.0
numpy              1.21.6
pip                22.0.4
pyglet             1.5.27
setuptools         47.1.0
typing_extensions  4.4.0
zipp               3.13.0
```


