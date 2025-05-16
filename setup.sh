#!/bin/bash
# Setup script for the transcript summarizer project
# Creates a virtual environment and installs dependencies

echo "Setting up the transcript summarizer environment..."

# Create a virtual environment
python3 -m venv venv
echo "Virtual environment created at ./venv"

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Download NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt')"

echo "Setup complete! To activate the environment, run:"
echo "source venv/bin/activate"
