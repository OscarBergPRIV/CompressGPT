#!/bin/bash
#SBATCH --job-name=accelerate_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-%j.out

# Load a base Python module (adjust version if needed)
module load python/3.10 || echo "⚠️ Could not load python module, continuing..."

# Path to your virtual environment
VENV_DIR=/home/oscar.berg/LM_test/lm

# Check if the venv's python binary works
if [ ! -x "$VENV_DIR/bin/python" ] || ! "$VENV_DIR/bin/python" -V &>/dev/null; then
    echo "⚠️ Virtual environment appears broken or missing. Recreating..."
    rm -rf "$VENV_DIR"
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    echo "⚠️ Virtual environment was recreated. Please reinstall your packages manually."
    echo "   You can do this on a login node using:"
    echo "       source $VENV_DIR/bin/activate"
    echo "       pip install <your packages>"
    exit 1
else
    echo "✅ Using existing virtual environment."
    source "$VENV_DIR/bin/activate"
fi

# Print environment info for debugging
echo "Python executable: $(which python)"
python --version

# Run your program
python /home/oscar.berg/compressGPT/extract_smashed_data.py
