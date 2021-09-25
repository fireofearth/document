#!/bin/bash

# APPROOT is the path of the carla-collect/ repository root directory
export APPROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set CARLA Simulatory directory manually
CARLANAME=carla-0.9.12
export CARLA_DIR=/home/$(whoami)/src/$CARLANAME

# Enable the Python environment
conda activate ml7

# Automatic path linking
export PYCARLA=$CARLA_DIR/PythonAPI/carla/dist/$CARLANAME-py3.7-linux-x86_64.egg
export UTILITY=$APPROOT/python-utility/utility
export CARLAUTIL=$APPROOT/python-utility/carlautil

# Setting Python path
export PYTHONPATH=$PYCARLA:$UTILITY:$CARLAUTIL:$APPROOT:$PYTHONPATH
export PYTHONPATH=$CARLA_DIR/PythonAPI/carla:$PYTHONPATH
