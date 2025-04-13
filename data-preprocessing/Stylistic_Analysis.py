#!/usr/bin/env python3
"""
STYLISTIC ANALYSIS 
Input: ../data/meetingBank_cleaned.csv.gz
Output: ../data/meetingBank_styled.csv.gz
"""

import os
import pandas as pd
import gzip
import spacy
from textblob import TextBlob
from tqdm import tqdm

# Initialize optimized NLP
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer", "tagger"])
nlp.add_pipe('sentencizer')
nlp.max_length = 2000000  # Handle long documents

def analyze_text(text):
    """Fast text analysis with error handling"""
    if not isinstance(text, str) or not text.strip():
        return {}
    
    try:
        doc = nlp(text)
        words = [t.text for t in doc if not t.is_punct]
        word_count = len(words)
        
        return {
            "word_count": word_count,
            "sentence_count": len(list(doc.sents)),
            "motion_count": text.lower().count("move") + text.lower().count("motion"),
            "avg_word_len": round(sum(len(w) for w in words)/max(1, word_count), 1),
            "sentiment": round(TextBlob(text[:5000]).sentiment.polarity, 2)  # Truncated for speed
        }
    except Exception as e:
        print(f"\n Error analyzing text: {str(e)}")
        return {}


# Get paths - goes up one level then into data/
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
input_path = os.path.join(parent_dir, "data", "meetingBank_cleaned.csv.gz")
output_path = os.path.join(parent_dir, "data", "meetingBank_styled.csv.gz")

# Verify paths

if not os.path.exists(input_path):
    print(f" Error: Input file not found")


# Load data
print(" Loading data...")
try:
    df = pd.read_csv(input_path, compression='gzip', header=None, names=["transcript"])
    print(f"Found {len(df)} records")
except Exception as e:
    print(f"Failed to load file: {str(e)}")


try:
    tqdm.pandas(desc="Processing")
    metrics = df["transcript"].progress_apply(analyze_text).apply(pd.Series)
    result = pd.concat([df, metrics], axis=1)
except KeyboardInterrupt:
    print(" Interrupted! Saving partial results...")
    metrics = df["transcript"].head(0).apply(analyze_text).apply(pd.Series)  # Empty df with columns
    result = pd.concat([df, metrics], axis=1)
except Exception as e:
    print(f" Analysis failed: {str(e)}")

# Save results
result.to_csv(output_path, index=False, compression="gzip")
print(f"Saved analyzed data to:\n{output_path}")
print("\nAdded metrics:", list(metrics.columns))

