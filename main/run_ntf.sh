#!/bin/bash

# Exit on error
set -e

# Activate conda environment
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate ntf_env

# Run the Python script
python $(dirname "$0")/ntf_app.py
