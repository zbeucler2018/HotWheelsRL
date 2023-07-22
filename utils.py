import os
import retro
import multiprocessing


def get_retro_install_path(verbose: bool = False) -> str:
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
    Returns true if in colab
    """
    try:
        import google.colab

        return True
    except Exception as err:
        return False


def get_num_cpus() -> int:
    """
    Returns number of cpus
    """
    return multiprocessing.cpu_count()