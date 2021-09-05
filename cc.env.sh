#!/bin/bash

export APPROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export UTILITY=$APPROOT/../python-utility/utility
export PYTHONPATH=$UTILITY:$PYTHONPATH

