#!/bin/bash

ENV_NAME="./conda_env"  
PYTHON_VERSION="3.11"
REQUIREMENTS_FILE="requirements.txt"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: Conda not found. Please install conda or add it to PATH."
    exit 1
fi

# Initialize conda for shell
eval "$(conda shell.bash hook)"

# Check if environment exists at path
if [ -d "$ENV_NAME" ] && conda env list | grep -q "$(pwd)/$ENV_NAME\s"; then
    echo "Environment '$ENV_NAME' exists. Activating..."
else
    echo "Environment '$ENV_NAME' not found. Creating with Python $PYTHON_VERSION..."
    
    # Accept ToS first if needed
    if ! conda create -y -p "$ENV_NAME" python="$PYTHON_VERSION" 2>/dev/null; then
        echo "Conda creation failed, likely due to ToS. Accepting Terms of Service..."
        conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
        conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
        conda create -y -p "$ENV_NAME" python="$PYTHON_VERSION"
    fi
fi

# Activate the environment
conda activate "$ENV_NAME"

# Verify activation
if [ $? -eq 0 ]; then
    echo "Successfully activated '$ENV_NAME'"
    echo "Python version: $(python --version 2>&1)"
    echo "Environment location: $(which python)"
else
    echo "Failed to activate environment '$ENV_NAME'"
    exit 1
fi

# Check and install requirements.txt if present
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Found $REQUIREMENTS_FILE. Installing packages..."
    pip install -r "$REQUIREMENTS_FILE"
    
    if [ $? -eq 0 ]; then
        echo "All packages from $REQUIREMENTS_FILE installed successfully"
    else
        echo "Warning: Some packages from $REQUIREMENTS_FILE failed to install"
    fi
else
    echo "No $REQUIREMENTS_FILE found. Skipping package installation."
fi

echo "Setup completed!"
echo "Conda environment created at: $(pwd)/$ENV_NAME"