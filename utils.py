import os
import multiprocessing
import argparse
import pprint
import retro
import enum
from stable_baselines3.common.policies import obs_as_tensor
import gymnasium as gym
import yaml
from typing import Any


class HotWheelsStates(str, enum.Enum):
    """
    Enviroments to put the agent into
    """

    DEFAULT = "TRex_Valley_single"
    TREX_VALLEY_SINGLE = "TRex_Valley_single"
    TREX_VALLEY_MULTI = "TRex_Valley_multi"
    DINO_BONEYARD_MULTI = "Dinosaur_Boneyard_multi"


def make_retro(
    *,
    game: str = "HotWheelsStuntTrackChallenge-gba",
    state: HotWheelsStates = HotWheelsStates.DEFAULT,
    render_mode="rgb_array",
    **kwargs,
):
    env = retro.make(
        game,
        state=f"{state}.state",
        info=retro.data.get_file_path(game, f"{state}.json"),
        render_mode=render_mode,
        **kwargs,
    )
    return env


def predict_action_prob(model, obs):
    """
    Returns the action probability
    of a obs
    https://stackoverflow.com/questions/66428307/how-to-get-action-propability-in-stable-baselines-3
    """
    _obs = obs_as_tensor(obs, model.policy.obs_to_tensor(obs)[0])
    dis = model.policy.get_distribution(_obs)
    probs = dis.distribution.probs
    probs_np = probs.detach().numpy()
    return probs_np


def get_retro_install_path() -> str:
    """
    returns the install filepath of [gym,stable]-retro
    """
    retro_directory = os.path.dirname(retro.__file__)
    return f"{retro_directory}/data/stable/"


def install_rom(game: str) -> None:
    """
    Installs the rom into retro with a link
    for easy editing
    """

    game_path = os.path.join(os.getcwd(), game)
    stable_integration_path = get_retro_install_path()

    assert os.path.isdir(game_path), f"{game_path} is not a valid directory."

    dest_path = os.path.join(stable_integration_path, game)

    if os.path.islink(dest_path):
        print(f"Removing existing symlink: {dest_path}")
        os.remove(dest_path)

    os.symlink(game_path, dest_path)
    print(f"Created symlink: {dest_path} -> {game_path}")

    os.system(f"python3 -m retro.import {game_path}")


def in_colab() -> bool:
    """
    Returns true if in colab notebook
    """
    return os.getenv("COLAB_RELEASE_TAG") is not None


def get_num_cpus() -> int:
    """
    Returns number of cpus
    """
    return multiprocessing.cpu_count()


def delete_files_containing_string(search_string):
    current_directory = os.getcwd()

    for filename in os.listdir(current_directory):
        if search_string in filename:
            file_path = os.path.join(current_directory, filename)
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")


def print_args(func: callable):
    """
    Decorator to print the keyword arguments
    of the function it decorates
    """

    def _wrapper(args):
        print("------------")
        for k, v in args.items():
            print(k, v)
        print("------------")
        return func(args)

    return _wrapper


class Config:
    def __init__(self, file_path: str = None):
        # RetroEnv
        self.game: str = "HotWheelsStuntTrackChallenge-gba"
        self.state: str = "Dinosaur_Boneyard_multi"
        self.scenario: str = None

        # Model
        self.total_steps: int = 20_000_000
        self.num_envs: int = 5
        self.resume: bool = False
        self.model_load_path: str = None
        self.run_id: str = None
        self.model_save_freq: int = 50_000
        self.model_save_path: str = "./models/models/"
        self.best_model_save_path: str = "./models/best_models/"
        self.gdrive_model_save_path: str = (
            "/content/gdrive/MyDrive/HotWheelsRL/data/models/"
        )
        self.gdrive_best_model_save_path: str = (
            "/content/gdrive/MyDrive/HotWheelsRL/data/best_models/"
        )

        # PPO model parameters
        self.policy: str = "CnnPolicy"
        self.learning_rate: Any = lambda f: f * 2.5e-4
        self.n_steps: int = 128
        self.batch_size: int = 32
        self.n_epochs: int = 4
        self.gamma: float = 0.99
        self.gae_lambda: float = 0.95
        self.clip_range: float = 0.1
        self.ent_coef: float = 0.01

        # Env / Wrappers
        self.action_space: list[list[str]] = [
            [],
            ["A"],
            ["UP"],
            ["DOWN"],
            ["LEFT"],
            ["RIGHT"],
            ["L", "R"],
        ]
        self.frame_skip: int = 4
        self.frame_skip_prob: float = 0.25
        self.terminate_on_crash: bool = True
        self.terminate_on_wall_crash: bool = True
        self.use_deepmind_env: bool = True
        self.max_episode_steps: int = 5_100
        self.frame_stack: int = 4
        self.trim_obs: bool = False
        self.minimap_obs: bool = False

        # Reward
        self.crash_reward: int = -5
        self.wall_crash_reward: int = -5

        # Evaluation
        self.evaluation_statename: str = "Dinosaur_Boneyard_multi"
        self.training_states: list[str] = [
            "Dinosaur_Boneyard_multi_71",
            "Dinosaur_Boneyard_multi_156",
            "Dinosaur_Boneyard_multi_180",
            "Dinosaur_Boneyard_multi_290",
            "Dinosaur_Boneyard_multi",
        ]
        self.eval_freq: int = max(200_000 // self.num_envs, 1)
        self.render_eval: bool = False

        # Misc
        self.skip_wandb: bool = False
        self.config_file: str = None
        self.in_colab: bool = in_colab()
        # self.tensorflow_log_path
        # self.wandb_log_path
        # self.eval_log_path

        if file_path:
            self.load_file_config(file_path)

        self.check_config_is_valid()

    def load_file_config(self, file_path):
        """
        loads a given config yaml file
        """
        with open(file_path, "r") as file:
            config_data = yaml.safe_load(file)

        # Create class variables for each field in the config file
        for key, value in config_data.items():
            if key == "eval_freq":
                setattr(self, key, max(value // self.num_envs, 1))
            setattr(self, key, value)

    def __repr__(self) -> str:
        """
        For a nicer print
        """
        properties = vars(self)
        formatted_properties = pprint.pformat(properties, sort_dicts=False)
        return formatted_properties

    def check_config_is_valid(self) -> None:
        """
        Throws exception if the config doesn't meet
        the rules below
        """
        # if resuming, make sure we have a defined
        # run_id, and model_load_path too
        if self.resume and not all([self.resume, self.run_id, self.model_load_path]):
            raise AssertionError(
                f"'resume' , 'run_id' , and 'model_load_path' must be defined to resume training a model."
            )

        # Ensure only 1 obs wrapper can be used at once
        if self.trim_obs and self.minimap_obs:
            raise AssertionError(
                f"Cannot use both 'trim_obs' and 'minimap_obs'. You can only use one obs wrapper at a time."
            )

        # Ensure that the amount of training states is equal
        # to the amount of envs
        if self.training_states and (len(self.training_states) != self.num_envs):
            raise AssertionError(
                f"The amount of training states ({len(self.training_states)}) must be the same as the amount of envs ({self.num_envs}) used for training."
            )
