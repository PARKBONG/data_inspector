#!/bin/bash

ENV_NAME="data_inspector_py311"
PYTHON_VERSION="3.11"

echo "=========================================================="
echo "[Conda Setup] Creating environment: $ENV_NAME (Python $PYTHON_VERSION)"
echo "=========================================================="

# Create conda environment
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
if [ $? -ne 0 ]; then
    echo "[ERROR] Conda environment creation failed."
    exit 1
fi

# Install requirements if file exists
if [ -f "requirements.txt" ]; then
    echo ""
    echo "[Pip Install] Installing dependencies from requirements.txt..."
    conda run -n "$ENV_NAME" pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "[ERROR] Pip installation failed."
        exit 1
    fi
else
    echo ""
    echo "[SKIP] requirements.txt not found. Skipping package installation."
fi

echo ""
echo "=========================================================="
echo "[SUCCESS] Environment '$ENV_NAME' is ready."
echo "To activate: conda activate $ENV_NAME"
echo "=========================================================="
