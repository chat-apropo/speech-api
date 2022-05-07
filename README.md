This is a minimal flask api with tts and stt endpoints. This runs on python 3.7.

This is pretty shitcoded so far.

# Installation

## Larynx 
https://github.com/rhasspy/larynx

Installing the requirements.txt on a virtual enviroment will do it.

## Coqui
https://github.com/coqui-ai/STT

Create a `models` folder and put the coqui models inside folders there. Create folders like `english` `romanian` with the models inside.

# Configuration

Create a `config.py` setting the `BEADER` http header and `PY_PATH` to the virtual env python path, like `/home/someone/project/venv/bin/python3`:. Edit the paths on `start.sh` accordingly.
