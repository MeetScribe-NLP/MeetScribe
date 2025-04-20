#!/bin/bash

set -e  

echo "Starting setup..."

# === Core dependencies ===
echo "Installing core Python packages..."
pip install --upgrade pip
pip install pandas torch transformers sentencepiece

# === NLP libraries ===
echo "Installing NLP libraries..."
pip install spacy textblob datasets

# === SpaCy English model ===
echo "Downloading SpaCy English model..."
python -m spacy download en_core_web_sm

# === Evaluation & audio tools ===
echo "Installing BERTScore and text-to-speech..."
pip install bert_score gTTS IPython

echo "All installations completed successfully."
