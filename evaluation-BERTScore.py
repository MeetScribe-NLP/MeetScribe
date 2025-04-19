import pandas as pd
import os
from bert_score import score

# Load your dataset
path = os.getcwd() + "/summarized-data/summaries_without_metadata.csv"
df = pd.read_csv(path)

# Replace NaNs with empty strings or drop rows with missing summaries/transcripts
df = df.dropna(subset=['transcript', 'textrank_summary', 'lexrank_summary', 'bertsum_summary'])

# Ensure all text inputs are strings
df['transcript'] = df['transcript'].astype(str)
df['textrank_summary'] = df['textrank_summary'].astype(str)
df['lexrank_summary'] = df['lexrank_summary'].astype(str)
df['bertsum_summary'] = df['bertsum_summary'].astype(str)

# Define a function to compute BERTScore
def compute_bertscore(reference_texts, candidate_texts, lang='en'):
    P, R, F1 = score(candidate_texts, reference_texts, lang=lang, verbose=True)
    return F1.tolist()

# Run the evaluation
df['bertscore_textrank'] = compute_bertscore(df['transcript'].tolist(), df['textrank_summary'].tolist())
df['bertscore_lexrank'] = compute_bertscore(df['transcript'].tolist(), df['lexrank_summary'].tolist())
df['bertscore_bertsum'] = compute_bertscore(df['transcript'].tolist(), df['bertsum_summary'].tolist())

# Save the result
df.to_csv("evaluated_dataset.csv", index=False)

