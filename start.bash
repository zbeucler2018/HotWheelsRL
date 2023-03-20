#!/bin/bash

set -e
build_first="$1"

# --build arg
if [ "$build_first" == "--build" ]; then
    docker build -t retro . --no-cache
fi

docker run -it --name retro_env -p 5555:5555 -v $(pwd):/home retro bash