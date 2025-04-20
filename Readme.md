# Meeting Summarization Pipeline

##  Setup Instructions

Before running any code in this repository, follow the steps below to set up your environment and install dependencies:

## (Optional) Create a Virtual Environment

python -m venv venv
source venv/bin/activate


## Install Required Dependencies
Make the setup script executable (if not already):
chmod +x setup.sh

Run the setup script:
./setup.sh

This will install all required Python packages and download SpaCy's English language model.

##  Project Overview
- Automates meeting summaries, saving 4+ hours per professional per week
- Captures critical decisions and action items with 100% phrasing accuracy
- Achieves 80% content reduction while preserving essential context
- Generates multi-format outputs (text and audio) for flexible consumption
- Removes human bias from manual note-taking

##  Project Workflow and File Descriptions
All summaries are generated with and without metadata. Due to the high volume, most data files are stored in compressed (.gz) format.

## 1. Data Collection
Script: load_data.py
Downloads meeting transcript data and stores it in: data/meetingBank.csv.gz

## 2. Data Cleaning
Script: data-preprocessing/clean_transcript.py
Removes noise and irrelevant content. Output stored in: data/meetingBank_cleaned.csv.gz

## 3. Stylistic Analysis
Script: data-preprocessing/Stylistic_Analysis.py
Performs stylistic feature extraction and stores results in: data/meetingBank_styled.csv.gz

## 4. Summarization
Scripts:
Extractive: extractive_summarization.py (TextRank, LexRank, BERTSum)
Abstractive: abstractive_summarization.py (T5, BART, OpenAI)

All generated summaries are stored in: summarized-data/

## 5. Audio Output
Script: play-audio.py
Generates audio for summaries. Currently supports a demo of the first OpenAI summary.
The audio can be played from: temp_audio.py

## 6. Summary Evaluation
Evaluated using four different metrics:
- Readability – Flesch-Kincaid Grade Level (FKGL)
- Quality – LLM-based evaluation using GPT
- Compression Ratio
- Semantic Similarity – BERTScore

Scripts:
- Abstractive:
  evaluation/evaluation_abstractive.py
  evaluation/evaluation_abstractive_BERTScore.py
- Extractive:
  evaluation/evaluation_extractive.py
  evaluation/evaluation_extractive_BERTScore.py

Evaluation results are stored in: evaluation/summaries_metrics/

## 7. Visualization & Analysis
Scripts:
- For FKGL, GPT evaluation, Compression: evaluation/evaluation_graphs.py
- For BERTScore: evaluation/evaluation_graphs_BERTScore.py

Generated graphs are stored in: evaluation/evaluation_graphs/