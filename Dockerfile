FROM python:3.8

WORKDIR /home

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y \
        ffmpeg libsm6 libxext6 \
        git \
        neovim

RUN pip install \
        git+https://github.com/DLR-RM/stable-baselines3@feat/gymnasium-support \
        git+https://github.com/Stable-Baselines-Team/stable-baselines3-contrib@feat/gymnasium-support \
        git+https://github.com/MatPoliquin/stable-retro.git \
        jupyter \
        notebook \
        matplotlib \
        opencv-python \
        moviepy

#ENTRYPOINT [ "jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root" ]