#!/bin/bash
export PATH=$PATH:/home/mattf/programs/STT/venv/bin
export WORKON_HOME=/home/mattf/programs/STT/venv
export VIRTUALENVWRAPPER_PYTHON=/home/mattf/programs/STT/venv/bin/python3
/home/mattf/programs/STT/venv/bin/gunicorn --workers 2 --bind 127.0.0.1:5555 -m 007 api:app
