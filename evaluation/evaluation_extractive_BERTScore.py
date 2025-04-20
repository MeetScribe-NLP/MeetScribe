import pandas as pd
import os
import zipfile
from bert_score import score

# === Set up base paths ===
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "..", "summarized-data")

# Input ZIP files
zip_files = {
    "with_metadata": os.path.join(data_dir, "summaries_with_metadata.csv.zip"),
    "without_metadata": os.path.join(data_dir, "summaries_without_metadata.csv.zip")
}

# === Function to load a CSV from a ZIP ===
def load_csv_from_zip(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as z:
        csv_files = [f for f in z.namelist() if f.endswith('.csv') and "__MACOSX" not in f]
        if not csv_files:
            raise FileNotFoundError(f"No valid CSV file found in {zip_file}")
        with z.open(csv_files[0]) as f:
            return pd.read_csv(f)

# === Function to run BERTScore on one dataset ===
def run_bertscore_evaluation(df):
    df = df.dropna(subset=['transcript', 'textrank_summary', 'lexrank_summary', 'bertsum_summary'])
    df['transcript'] = df['transcript'].astype(str)
    df['textrank_summary'] = df['textrank_summary'].astype(str)
    df['lexrank_summary'] = df['lexrank_summary'].astype(str)
    df['bertsum_summary'] = df['bertsum_summary'].astype(str)

    def compute(refs, cands):
        P, R, F1 = score(cands, refs, lang='en', verbose=True)
        return F1.tolist()

    print("Running BERTScore for TextRank...")
    df['bertscore_textrank'] = compute(df['transcript'].tolist(), df['textrank_summary'].tolist())

    print("Running BERTScore for LexRank...")
    df['bertscore_lexrank'] = compute(df['transcript'].tolist(), df['lexrank_summary'].tolist())

    print("Running BERTScore for BertSUM...")
    df['bertscore_bertsum'] = compute(df['transcript'].tolist(), df['bertsum_summary'].tolist())

    return df

# === Process both datasets ===
for label, zip_path in zip_files.items():
    print(f"\n Processing: {label.replace('_', ' ').title()}")
    df = load_csv_from_zip(zip_path)
    evaluated_df = run_bertscore_evaluation(df)
    
    output_filename = f"evaluated_dataset_{label}.csv"
    output_path = os.path.join(current_dir, "summaries_metrics", output_filename)
    evaluated_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
