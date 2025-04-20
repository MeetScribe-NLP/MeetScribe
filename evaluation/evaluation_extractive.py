import pandas as pd
import os
import textstat
import zipfile
from openai import OpenAI
import tiktoken
import json

# === Load OpenAI API key and instantiate client
current_dir = os.path.dirname(os.path.abspath(__file__))
key_path = os.path.join(current_dir, "..", "OPENAI_API_KEY")
if not os.path.isfile(key_path):
    raise FileNotFoundError(f"API key file not found: {key_path}")
with open(key_path, "r") as f:
    client = OpenAI(api_key=f.read().strip())

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

# === Tokenizer for chunking prompts
model_name           = "gpt-4o-mini"
encoding             = tiktoken.encoding_for_model(model_name)
MAX_TOKENS_PER_CHUNK = 16000
EVAL_MAX_TOKENS      = 50
OVERHEAD_TOKENS      = 200  # reserve for system+formatting

def chunk_text(text: str, max_tokens: int):
    tokens = encoding.encode(text)
    for i in range(0, len(tokens), max_tokens):
        yield encoding.decode(tokens[i: i + max_tokens])

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
        print(f"\nEvaluating {row}...")
        original = row['transcript']
        summary = row[summary_col]

        if pd.isna(summary) or pd.isna(original):
            for k in metrics:
                metrics[k].append(None)
            continue

        try:
            fkgl = textstat.flesch_kincaid_grade(summary)
            orig_len = len(original.split())
            summ_len = len(summary.split())
            comp_ratio = summ_len / orig_len if orig_len > 0 else None
            # 3. Chunk original & summary
            max_chunk = MAX_TOKENS_PER_CHUNK - OVERHEAD_TOKENS - EVAL_MAX_TOKENS
            orig_chunks = list(chunk_text(original, max_chunk))
            summ_chunks = list(chunk_text(summary, max_chunk))

            # 4. Evaluate each chunk-pair using GPT
            chunk_scores = []
            for o_chunk, s_chunk in zip(orig_chunks, summ_chunks):
                prompt = (
                    "Original:\n" + o_chunk + "\n\n"
                    "Summary:\n" + s_chunk + "\n\n"
                    "Rate this summary 1–5 for coherence, coverage, fluency. "
                    "Reply as JSON: {\"coherence\":int, \"coverage\":int, \"fluency\":int}."
                )
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert summary evaluator."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=EVAL_MAX_TOKENS
                )
                try:
                    scores = json.loads(resp.choices[0].message.content.strip())
                except:
                    scores = {"coherence": None, "coverage": None, "fluency": None}
                chunk_scores.append(scores)

            # 5. Average chunk scores
            def avg(key):
                vals = [c.get(key) for c in chunk_scores if isinstance(c.get(key), (int, float))]
                return sum(vals)/len(vals) if vals else None

            # Append computed metrics
            metrics['fkgl'].append(fkgl)
            metrics['compression_ratio'].append(comp_ratio)
            metrics['coherence'].append(avg("coherence"))
            metrics['coverage'].append(avg("coverage"))
            metrics['fluency'].append(avg("fluency"))

        except:
            for k in metrics:
                metrics[k].append(None)

    return pd.DataFrame(metrics)

# === Load CSVs from parent directory ===
df_with = load_csv_from_zip(with_zip).head(501)
df_without = load_csv_from_zip(without_zip).head(501)

# === Save 6 metrics CSVs ===
for col, model in models.items():
    df_meta_metrics = calculate_metrics(df_with, col)
    df_meta_metrics.to_csv(os.path.join(evaluation_data_folder, f"{model}_metadata_summaries_metrics.csv"), index=False)

    df_no_meta_metrics = calculate_metrics(df_without, col)
    df_no_meta_metrics.to_csv(os.path.join(evaluation_data_folder, f"{model}_transcript_summaries_metrics.csv"), index=False)

print("All 6 metrics CSVs generated in:", evaluation_data_folder)
