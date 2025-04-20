import os
import json
import pandas as pd
import textstat
from openai import OpenAI
import tiktoken
import zipfile

# 1. Load OpenAI API key and instantiate client
current_dir = os.path.dirname(os.path.abspath(__file__))
key_path = os.path.join(current_dir, "..", "OPENAI_API_KEY")
if not os.path.isfile(key_path):
    raise FileNotFoundError(f"API key file not found: {key_path}")
with open(key_path, "r") as f:
    client = OpenAI(api_key=f.read().strip())

# 2. Paths & load original transcripts
summarized_data_folder = os.path.join(current_dir, "..", "summarized-data")
data_folder = os.path.join(current_dir, "..", "data")
evaluation_data_folder = os.path.join(current_dir, "summaries_metrics")
input_csv_path = os.path.join(data_folder, "meetingBank_styled.csv.gz")
df_orig = pd.read_csv(input_csv_path, compression="gzip")

# 3. Map summary files
summary_files = {
    "t5_transcript_summaries":     "t5_transcript_summaries.csv.gz",
    "bart_transcript_summaries":   "bart_transcript_summaries.csv.gz",
    "openai_transcript_summaries": "openai_transcript_summaries.csv.gz",
    "t5_metadata_summaries":       "t5_metadata_summaries.csv.gz",
    "bart_metadata_summaries":     "bart_metadata_summaries.csv.gz",
    "openai_metadata_summaries":   "openai_metadata_summaries.csv.gz",
}

# 4. Tokenizer for chunking prompts
model_name           = "gpt-4o-mini"
encoding             = tiktoken.encoding_for_model(model_name)
MAX_TOKENS_PER_CHUNK = 16000
EVAL_MAX_TOKENS      = 50
OVERHEAD_TOKENS      = 200  # reserve for system+formatting

def chunk_text(text: str, max_tokens: int):
    tokens = encoding.encode(text)
    for i in range(0, len(tokens), max_tokens):
        yield encoding.decode(tokens[i: i + max_tokens])

# We'll store each metrics DataFrame for later ranking
metrics_map = {}

# 5. Evaluation loop
for name, fname in summary_files.items():
    print(f"\nEvaluating {name}...")
    sum_df = pd.read_csv(os.path.join(summarized_data_folder, fname), compression="gzip")

    # choose reference text
    if "transcript" in name:
        originals = df_orig["transcript"].fillna("").tolist()
    else:
        originals = (
            "Transcript: " + df_orig["transcript"].fillna("") + "\n" +
            "Word Count: "   + df_orig["word_count"].astype(str) + "\n" +
            "Sentence Count:" + df_orig["sentence_count"].astype(str) + "\n" +
            "Motion Count: "  + df_orig["motion_count"].astype(str) + "\n" +
            "Avg Word Len: "  + df_orig["avg_word_len"].astype(str) + "\n" +
            "Sentiment: "     + df_orig["sentiment"].astype(str)
        ).tolist()

    metrics = []
    for orig, summ in zip(originals, sum_df["summary"].fillna("")):
        # 1. Readability (Flesch-Kincaid Grade Level)
        fkgl = textstat.flesch_kincaid_grade(summ)

        # 2. Compression ratio
        orig_words = len(orig.split())
        summ_words = len(summ.split())
        comp_ratio = summ_words / orig_words if orig_words > 0 else None

        # 3. Chunk original & summary
        max_chunk = MAX_TOKENS_PER_CHUNK - OVERHEAD_TOKENS - EVAL_MAX_TOKENS
        orig_chunks = list(chunk_text(orig, max_chunk))
        summ_chunks = list(chunk_text(summ, max_chunk))

        # 4. Evaluate each chunk-pair
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
                    {"role":"system","content":"You are an expert summary evaluator."},
                    {"role":"user",  "content":prompt}
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

        metrics.append({
            "fkgl": fkgl,
            "compression_ratio": comp_ratio,
            "coherence": avg("coherence"),
            "coverage": avg("coverage"),
            "fluency": avg("fluency"),
        })

    # 6. Save metrics
    metrics_df = pd.DataFrame(metrics)
    out_path   = os.path.join(evaluation_data_folder, f"{name}_metrics.csv")
    metrics_df.to_csv(out_path, index=False)
    metrics_map[name] = metrics_df
    print(f"Saved metrics to {out_path}")
