#!/bin/bash

# Simple test runner script

echo "Running Calibration Toolbox tests..."
echo "======================================"
echo ""

# Run tests without coverage (for when pytest-cov is not installed)
python -m pytest tests/ -v

# If you want coverage, install pytest-cov first:
# pip install pytest-cov
# Then run: pytest tests/ -v --cov=calibration_toolbox --cov-report=html
