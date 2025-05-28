#!/bin/bash

# Exit on error
set -e

# Set venv directory
VENV_DIR="venv"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Install requirements
pip install -r requirements.txt

# Run the UI
python UI.py