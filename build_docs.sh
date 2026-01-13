#!/bin/bash

# Build documentation website locally

echo "Building Calibration Toolbox Documentation Website..."
echo "======================================================="
echo ""

# Install Sphinx if needed
echo "Installing documentation dependencies..."
pip install sphinx sphinx-rtd-theme nbsphinx -q

# Build HTML documentation
echo ""
echo "Building HTML documentation..."
cd docs
make html

echo ""
echo "✓ Documentation built successfully!"
echo ""
echo "To view the website, open: docs/_build/html/index.html"
echo "Or run: python -m http.server 8000 --directory docs/_build/html"
