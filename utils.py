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


