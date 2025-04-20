import pandas as pd
import os
import textstat
import zipfile

# === Setup path to summarized-data folder in parent directory ===
current_dir = os.path.dirname(os.path.abspath(__file__))
evaluation_data_folder = os.path.join(current_dir, "summaries_metrics")
summarized_data_folder = os.path.join(current_dir, "..", "summarized-data")

# === Load CSV cleanly from ZIP (ignores __MACOSX files) ===
def load_csv_from_zip(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        for f in z.namelist():
            if f.endswith(".csv") and "__MACOSX" not in f:
                return pd.read_csv(z.open(f))
    raise ValueError(f"No valid CSV found in {zip_path}")

# === Input zip files (relative to parent folder) ===
with_zip = os.path.join(summarized_data_folder, "summaries_with_metadata.csv.zip")
without_zip = os.path.join(summarized_data_folder, "summaries_without_metadata.csv.zip")

# === Model mapping ===
models = {
    'textrank_summary': 'textrank',
    'lexrank_summary': 'lexrank',
    'bertsum_summary': 'bertsum'
}

# === Metric calculator ===
def calculate_metrics(df, summary_col):
    metrics = {
        'fkgl': [],
        'compression_ratio': [],
        'coherence': [],
        'coverage': [],
        'fluency': []
    }

    for _, row in df.iterrows():
        original = row['transcript']
        summary = row[summary_col]

        if pd.isna(summary) or pd.isna(original):
            for k in metrics:
                metrics[k].append(None)
            continue

        try:
            metrics['fkgl'].append(textstat.flesch_kincaid_grade(summary))
            orig_len = len(original.split())
            summ_len = len(summary.split())
            metrics['compression_ratio'].append(round(summ_len / orig_len, 4) if orig_len > 0 else None)
            metrics['coherence'].append(len(summary.split('.')) / 5)  # Fake proxy
            metrics['coverage'].append(min(1.0, len(set(summary.split()) & set(original.split())) / 20))
            metrics['fluency'].append(min(1.0, 1 - abs(textstat.flesch_reading_ease(summary) - 60) / 100))
        except:
            for k in metrics:
                metrics[k].append(None)

    return pd.DataFrame(metrics)

# === Load CSVs from parent directory ===
df_with = load_csv_from_zip(with_zip)
df_without = load_csv_from_zip(without_zip)

# === Save 6 metrics CSVs ===
for col, model in models.items():
    df_meta_metrics = calculate_metrics(df_with, col)
    df_meta_metrics.to_csv(os.path.join(evaluation_data_folder, f"{model}_metadata_summaries_metrics.csv"), index=False)

    df_no_meta_metrics = calculate_metrics(df_without, col)
    df_no_meta_metrics.to_csv(os.path.join(evaluation_data_folder, f"{model}_transcript_summaries_metrics.csv"), index=False)

print("All 6 metrics CSVs generated in:", evaluation_data_folder)
