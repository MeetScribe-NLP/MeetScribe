import os
import pandas as pd
import torch
import zipfile
from openai import OpenAI
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    BartTokenizer, BartForConditionalGeneration
)
import tiktoken

# 1. Load OpenAI API key from file and instantiate client
current_dir = os.path.dirname(os.path.abspath(__file__))
key_path = os.path.join(current_dir, "OPENAI_API_KEY")
if not os.path.isfile(key_path):
    raise FileNotFoundError(f"API key file not found: {key_path}")
with open(key_path, "r") as f:
    api_key = f.read().strip()
client = OpenAI(api_key=api_key)

# 2. Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 3. Paths
data_folder    = os.path.join(current_dir, "data")
input_csv_path = os.path.join(data_folder, "meetingBank_styled.csv.gz")

# 4. Load data
df = pd.read_csv(input_csv_path, compression="gzip")
required_cols = ['transcript','word_count','sentence_count','motion_count','avg_word_len','sentiment']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# 5. Initialize T5 & BART
t5_tokenizer   = T5Tokenizer.from_pretrained("t5-small")
t5_model       = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
bart_model     = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)

# 6. Initialize tiktoken for chunking
encoding = tiktoken.encoding_for_model("gpt-4o-mini")
MAX_TOKENS_PER_CHUNK = 16000
SUMMARY_TOKENS = 150

def chunk_text(text: str, max_tokens: int):
    tokens = encoding.encode(text)
    for i in range(0, len(tokens), max_tokens):
        yield encoding.decode(tokens[i: i + max_tokens])

def summarize_openai_long(text: str, model="gpt-4o-mini"):
    # 1) split into chunks
    summaries = []
    for chunk in chunk_text(text, MAX_TOKENS_PER_CHUNK):
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert abstractive summarizer."},
                {"role": "user",   "content": f"Summarize the following text:\n\n{chunk}"}
            ],
            temperature=0.7,
            max_tokens=SUMMARY_TOKENS
        )
        summaries.append(resp.choices[0].message.content.strip())
    # 2) if only one chunk, return it
    if len(summaries) == 1:
        return summaries[0]
    # 3) otherwise summarize the concatenated summaries
    joined = "\n\n".join(summaries)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert abstractive summarizer."},
            {"role": "user",   "content": f"Summarize the following summaries:\n\n{joined}"}
        ],
        temperature=0.7,
        max_tokens=SUMMARY_TOKENS
    )
    return resp.choices[0].message.content.strip()

# 7. Summarization functions for local models
def summarize_t5(text, max_length=150, min_length=40):
    ids = t5_tokenizer.encode("summarize: " + text, return_tensors="pt", truncation=True).to(device)
    out = t5_model.generate(
        ids,
        max_length=max_length, min_length=min_length,
        length_penalty=2.0, num_beams=4, early_stopping=True
    )
    return t5_tokenizer.decode(out[0], skip_special_tokens=True)

def summarize_bart(text, max_length=150, min_length=40):
    ids = bart_tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True).to(device)
    out = bart_model.generate(
        ids,
        max_length=max_length, min_length=min_length,
        length_penalty=2.0, num_beams=4, early_stopping=True
    )
    return bart_tokenizer.decode(out[0], skip_special_tokens=True)

# 8. Prepare storage
t5_trans, bart_trans, openai_trans = [], [], []
t5_full, bart_full, openai_full     = [], [], []

# 9. Iterate and summarize each row
for idx, row in df.iterrows():
    txt = row['transcript'] or ""
    if not txt.strip():
        t5_trans.append(""); bart_trans.append(""); openai_trans.append("")
        t5_full.append(""); bart_full.append(""); openai_full.append("")
        continue

    print(f"Processing row {idx}...")

    # transcript-only
    t5_tr    = summarize_t5(txt)
    bart_tr  = summarize_bart(txt)
    oai_tr   = summarize_openai_long(txt)

    # all-fields combined
    combo = (
        f"Transcript: {row['transcript']}\n"
        f"Word Count: {row['word_count']}\n"
        f"Sentence Count: {row['sentence_count']}\n"
        f"Motion Count: {row['motion_count']}\n"
        f"Average Word Length: {row['avg_word_len']}\n"
        f"Sentiment: {row['sentiment']}"
    )
    t5_al    = summarize_t5(combo)
    bart_al  = summarize_bart(combo)
    oai_al   = summarize_openai_long(combo)

    t5_trans.append(t5_tr)
    bart_trans.append(bart_tr)
    openai_trans.append(oai_tr)
    t5_full.append(t5_al)
    bart_full.append(bart_al)
    openai_full.append(oai_al)

    if(idx == 500):
        break

# 10. Save all summaries
outputs = {
    "t5_transcript_summaries":     t5_trans,
    "bart_transcript_summaries":   bart_trans,
    "openai_transcript_summaries": openai_trans,
    "t5_metadata_summaries":       t5_full,
    "bart_metadata_summaries":     bart_full,
    "openai_metadata_summaries":   openai_full,
}

for name, arr in outputs.items():
    df_out = pd.DataFrame({"summary": arr})
    path   = os.path.join(data_folder, f"{name}.csv.gz")
    df_out.to_csv(path, index=False, compression="gzip")
    print(f"Saved {name} to {path}")
