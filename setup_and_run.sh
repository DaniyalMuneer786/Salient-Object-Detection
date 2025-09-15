#!/bin/bash

# Exit on error
set -e

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Conda first."
    exit 1
fi

# Check if environment exists
ENV_NAME="selfmask"
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment '$ENV_NAME' already exists. Activating..."
else
    echo "Creating conda environment '$ENV_NAME'..."
    conda create -n $ENV_NAME python=3.8 -y
fi

# Activate environment
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Check if PyTorch is installed
if ! python -c "import torch" &> /dev/null; then
    echo "Installing PyTorch..."
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
else
    echo "PyTorch is already installed."
fi

# Install dependencies from requirements.txt if it exists
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Installing core packages..."
    pip install wandb timm scipy numpy opencv-python matplotlib tqdm scikit-learn pillow
fi

# Make main.py executable if it exists
if [ -f "main.py" ]; then
    chmod +x main.py
fi

# Run the project
echo "Running the project..."
if [ -f "main.py" ]; then
    python main.py
else
    echo "main.py not found in the current directory."
    exit 1
fi

echo "Setup and execution completed!" 