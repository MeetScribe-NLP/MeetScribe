#!/bin/bash

# Install required Python packages
echo "Installing datasets..."
pip install datasets

echo "Installing spacy..."
pip install spacy

echo "Installing textblob..."
pip install textblob

# Download SpaCy's English model
echo "Downloading SpaCy English model..."
python -m spacy download en_core_web_sm

echo "All installations completed."
