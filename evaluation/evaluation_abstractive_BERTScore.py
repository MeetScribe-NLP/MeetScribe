import pandas as pd
import os
from bert_score import score

# === Setup paths ===
current_dir = os.path.dirname(os.path.abspath(__file__))
summaries_dir = os.path.join(current_dir, "..", "summarized-data")
transcript_path = os.path.join(current_dir, "..", "data", "meetingBank.csv.gz")

# === Summary files grouped by type ===
summary_files = {
    "with_metadata": {
        "t5": "t5_metadata_summaries.csv.gz",
        "bart": "bart_metadata_summaries.csv.gz",
        "openai": "openai_metadata_summaries.csv.gz"
    },
    "without_metadata": {
        "t5": "t5_transcript_summaries.csv.gz",
        "bart": "bart_transcript_summaries.csv.gz",
        "openai": "openai_transcript_summaries.csv.gz"
    }
}

# === Load reference transcripts ===
print("Loading reference transcripts...")
df_transcript = pd.read_csv(transcript_path, compression="gzip").head(501)
reference_texts = df_transcript["transcript"].astype(str).tolist()

# === Compute per-row BERTScore F1 ===
def compute_bertscore_per_row(summary_file_path, reference_texts):
    df_summary = pd.read_csv(summary_file_path, compression="gzip")
    candidate_texts = df_summary.iloc[:, 0].astype(str).tolist()
    
    if len(candidate_texts) != len(reference_texts):
        raise ValueError(f"Mismatch: {len(candidate_texts)} summaries vs {len(reference_texts)} transcripts")

    print(f"Computing BERTScore for: {os.path.basename(summary_file_path)}")
    _, _, F1 = score(candidate_texts, reference_texts, lang='en', verbose=True)
    return F1.tolist()

# === Process each group and save ===
for group, model_dict in summary_files.items():
    output_df = pd.DataFrame()

    for model, filename in model_dict.items():
        file_path = os.path.join(summaries_dir, filename)
        f1_scores = compute_bertscore_per_row(file_path, reference_texts)
        output_df[model] = f1_scores

    # Save result
    output_file = f"bertscore_abstractive_{group}.csv"
    output_path = os.path.join(current_dir, "summaries_metrics" , output_file)
    output_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
