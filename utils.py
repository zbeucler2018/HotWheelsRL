import os
import multiprocessing
import argparse
import retro
import enum
from stable_baselines3.common.policies import obs_as_tensor
import gymnasium as gym
from dataclasses import dataclass
from pprint import pprint


class HotWheelsStates(str, enum.Enum):
    """
    Enviroments to put the agent into
    """

    DEFAULT = "TRex_Valley_single"
    TREX_VALLEY_SINGLE = "TRex_Valley_single"
    TREX_VALLEY_MULTI = "TRex_Valley_multi"
    DINO_BONEYARD_MULTI = "Dinosaur_Boneyard_multi"


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


def print_args(func: callable):
    """
    Decorator to print the keyword arguments
    of the function it decorates
    """

    def _wrapper(args):
        print("------------")
        attrs = vars(args)
        pprint(attrs)
        print("------------")
        return func(args)

    return _wrapper


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


@dataclass
class Config:
    game: str
    state: str
    scenario: str
    total_steps: int
    num_envs: int
    resume: bool
    run_id: str
    model_path: str
    trim_obs: bool
    minimap_obs: bool
    evaluation_statename: str
    training_states: [str]

    def pretty_print(self):
        pass


def parse_args(parser: argparse.ArgumentParser) -> Config:
    """
    Parses arguments for CLI scripts
    """
    parser.add_argument("--game", default="HotWheelsStuntTrackChallenge-gba")
    parser.add_argument("--state", default=HotWheelsStates.DEFAULT, type=str)
    parser.add_argument("--scenario", default=None)
    parser.add_argument(
        "--total_steps", help="Total steps to train", type=int, required=True
    )
    parser.add_argument(
        "--num_envs",
        help="Number of envs to train at the same time. Default is 8",
        type=int,
        required=False,
        default=8,
    )

    parser.add_argument("--resume", help="Resume training a model", action="store_true")
    parser.add_argument(
        "--run_id", help="Wandb run ID to resume training a model", type=str
    )
    parser.add_argument(
        "--model_path", help="Path to saved model to resume training", type=str
    )

    parser.add_argument(
        "--trim_obs",
        help="Crop the observation such that the lap/race timers, speed dial, and minimap are not shown",
        action="store_true",
    )
    parser.add_argument(
        "--minimap_obs",
        help="Crop the observation so the model is given only the minimap",
        action="store_true",
    )

    parser.add_argument(
        "--evaluation_state", help="Statename to use for evaluation", type=str
    )

    parser.add_argument(
        "--training_states",
        help="Substates of the game state. These states must share the same data.json as the gamestate",
        nargs="*",
    )

    _args = parser.parse_args()

    # check for illegal resume options
    if (_args.resume or _args.run_id or _args.model_path) and not all(
        [_args.resume, _args.run_id, _args.model_path]
    ):
        raise Exception(
            f"--resume , --run_id , and --model_path must be defined to resume training"
        )

    # check for illegal obs options
    if sum([_args.trim_obs, _args.minimap_obs]) > 1:
        raise Exception(
            f"Only one obs flag (--trim_obs, --minimap_obs) can be used at a time"
        )

    # check there is 1 training state per env
    if not _args.training_states is None:
        if len(_args.training_states) != _args.num_envs:
            raise Exception(
                f"The amount of training states ({len(_args.training_states)}) must be equal to the amount of envs ({_args.num_envs})"
            )
    return _args
