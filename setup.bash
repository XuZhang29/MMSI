#!/bin/bash

# Exit on error
set -e

# Create and activate a virtual environment
echo "Creating virtual environment..."
python3 -m venv env
source env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Verify installation
echo "Verifying installation..."
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import transformers; print('Transformers version:', transformers.__version__)"
python -c "import numpy; print('NumPy version:', numpy.__version__)"

echo "Setup complete. To activate the virtual environment, run:"
echo "source env/bin/activate"

python train.py
