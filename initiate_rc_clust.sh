#!/bin/bash

# Load the Anaconda3 module
module load anaconda3/2021.05

# Initialize Conda in the current shell
conda init

# Activate the environment with PyTorch
source activate mote_torch

# Run a Python script to check if CUDA (GPU support) is available
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Check if a GPU is being used
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "GPU is being used."
else
    echo "GPU is NOT being used."
fi