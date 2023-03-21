import sys
import subprocess
import os
import shutil
import site
import importlib

def print_args_table(args):
  """
  Creates a table to show cli parameters
  """
  # Create table header
  table_header = ['Argument', 'Value']

  # Create table rows
  table_rows = []
  for arg in vars(args):
    table_rows.append([f"--{arg}", str(getattr(args, arg))])

  # Print table
  print(f"{table_header[0]:<20} {table_header[1]:<10}")
  print("-" * 30)
  for row in table_rows:
    print(f"{row[0]:<20} {row[1]:<10}")


def is_colab_notebook():
  """
  Returns True if the script is in a colab notebook
  """
  return 'sys.google.colab' in sys.modules


def gba_file_exists():
  """
  checks if the rom.gba file is in rom/
  """
  source_path = os.path.join(os.getcwd(), 'rom')
  rom_path = os.path.join(source_path, 'rom.gba')
  if not os.path.exists(rom_path):
      print(f"{rom_path} not found in {source_path}")
      return False
  return True



def get_package_location(package_name):
    """
    Returns the installation location of a pip package.
    """
    try:
        importlib.import_module(package_name)
        package_location = site.getsitepackages()
        package_location = [p for p in package_location if package_name in p][0]
        return package_location
    except ImportError:
        print(f"Package '{package_name}' is not installed.")

  


def import_rom():

  """
  imports rom into retro (wip)
      - should automatically detect if in colab or not


  - check if rom.gba is in rom/
    - if gdrive then import it
  - link rom/ to python3.8/dist-packages/retro/data/stable
  - run import module
  """
        
        
  # if is_colab_notebook():
  #   drive.mount('/content/gdrive')
  #   source = "/content/gdrive/MyDrive/theLab_/HotWheelsRL/rom.gba"
  #   dest = "/content/HotWheelsRL/rom/rom.gba"
  #   try:
  #     shutil.copy(source, dest)
  #   except Exception as err:
  #     raise Exception(err)
  # else:

  # folder_name="/content/HotWheelsRL/rom"
  # link_name="HotWheelsStuntTrackChallenge-gba"
  # lib_path="/usr/local/lib/python3.8/dist-packages/retro/data/stable"
  # # Validate input and handle errors
  # if not Path(folder_name).is_dir():
  #     raise ValueError(f"{folder_name} is not a valid directory.")
  # if not Path(lib_path).is_dir():
  #     raise ValueError(f"{lib_path} is not a valid directory.")
  # # Define paths as Path objects
  # source_path = Path.cwd() / folder_name
  # dest_path = Path(lib_path) / link_name
  # # Use Path.symlink_to() to create the symbolic link
  # try:
  #     dest_path.symlink_to(source_path)
  #     print(f"Created symlink: {dest_path} -> {source_path}")
  # except OSError as e:
  #     print(f"Error creating symlink: {e}")

      
  # os.system('python -m retro.import /content/HotWheelsRL/rom/')
  # return






  # source_path = os.path.join(os.getcwd(), 'rom')
  # link_name = 'HotWheelsStuntTrackChallenge-gba'
  # lib_path = '/usr/local/lib/python3.8/site-packages/retro/data/stable'

  # # check both dirs 
  # if not os.path.isdir(source_path):
  #     print(f"{source_path} is not a valid directory.")
  #     sys.exit(1)

  # if not os.path.isdir(lib_path):
  #     print(f"{lib_path} is not a valid directory.")
  #     sys.exit(1)

  # dest_path = os.path.join(lib_path, link_name)

  # if os.path.islink(dest_path):
  #     print(f"Removing existing symlink: {dest_path}")
  #     os.remove(dest_path)

  # if is_colab_notebook():
     

  # os.symlink(source_path, dest_path)
  # print(f"Created symlink: {dest_path} -> {source_path}")

  # # import into the library
  # subprocess.run(['python', '-m', 'retro.import', source_path], check=True)








