#!/bin/bash

# Optional: Print environment info
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"

# Install detectron2 manually
pip install git+https://github.com/facebookresearch/detectron2.git@v0.6
