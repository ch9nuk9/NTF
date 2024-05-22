#!/bin/bash

# Create a new conda environment
conda create -n NTF python=3.9 -y

# Activate the environment
source activate NTF

# Install dependencies
conda install pandas numpy matplotlib scipy tk -y

echo "Environment setup complete. To activate the environment, use 'conda activate NTF'."

# Use the following line of code to run this script prior running the NTF_main.py (Fresh setups only)
# bash setup_environment.sh
