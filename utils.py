import os

def get_filename(directory_path: str, extension: str):
    for filename in os.listdir(directory_path):
        if filename.endswith(extension):
            return filename
    return None

def find_file_with_extension(directory, extension):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                return os.path.abspath(os.path.join(root, file))
    return None