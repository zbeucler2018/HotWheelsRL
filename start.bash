#!/bin/bash

set -e
build_first="$1"

# --build arg
if [ "$build_first" == "--build" ]; then
    docker build -t retro . --no-cache
fi

docker run -it --rm --name retro_env -p 8888:8888 -p 5555:5555 -v $(pwd):/home retro