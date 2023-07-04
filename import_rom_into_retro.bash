#!/bin/bash

# By: Zack Beucler
# Integrates the game into the gym-retro library.

set -e

source_path="$(pwd)/HotWheelsStuntTrackChallenge-gba"
link_name="HotWheelsStuntTrackChallenge-gba"
#lib_path="/usr/local/lib/python3.8/site-packages/retro/data/stable"
lib_path="env/lib/python3.10/site-packages/retro/data/stable" # if working with a venv

if [ ! -d "$source_path" ]; then
    echo "$source_path is not a valid directory."
    exit 1
fi

if [ ! -d "$lib_path" ]; then
    echo "$lib_path is not a valid directory."
    exit 1
fi

dest_path="$lib_path/$link_name"

if [ -L "$dest_path" ]; then
    echo "Removing existing symlink: $dest_path"
    rm "$dest_path"
fi

ln -s "$source_path" "$dest_path"
echo "Created symlink: $dest_path -> $source_path"

# import into the library
python3 -m retro.import "$source_path"