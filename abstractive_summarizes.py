import os
import pandas as pd
import torch
import zipfile
from transformers import T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration

# Check if GPU is available and set device accordingly.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Build the path to the data folder and file.
current_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(current_dir, "data")
data_file_path = os.path.join(data_folder, "meetingBank_styled.csv.gz")

# Load the compressed CSV file.
df = pd.read_csv(data_file_path, compression="gzip")
print("CSV columns found:", df.columns.tolist())

# Check that required columns exist for the full summary.
required_columns = ['transcript', 'word_count', 'sentence_count', 'motion_count', 'avg_word_len', 'sentiment']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in the CSV file.")

# ----- Initialize the T5 Model and Tokenizer -----
t5_model_name = "t5-small"
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name).to(device)

# ----- Initialize the BART Model and Tokenizer -----
bart_model_name = "facebook/bart-large-cnn"
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name).to(device)

def summarize_t5(text, max_length=150, min_length=40):
    """
    Generate a summary for the given text using the T5 model.
    """
    input_text = "summarize: " + text.strip()  # T5 requires a task-specific prefix.
    input_ids = t5_tokenizer.encode(input_text, return_tensors="pt", truncation=True).to(device)
    
    summary_ids = t5_model.generate(
        input_ids,
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def summarize_bart(text, max_length=150, min_length=40):
    """
    Generate a summary for the given text using the BART model.
    """
    inputs = bart_tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True).to(device)
    
    summary_ids = bart_model.generate(
        inputs,
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Initialize lists to store summaries for all transcripts.
t5_transcript_summaries = []
bart_transcript_summaries = []
t5_all_summaries = []
bart_all_summaries = []

# Iterate over every transcript in the dataframe.
for idx, row in df.iterrows():
    transcript = row['transcript']
    if not isinstance(transcript, str) or not transcript.strip():
        # If the transcript is empty or invalid, store empty strings.
        t5_transcript_summaries.append("")
        bart_transcript_summaries.append("")
        t5_all_summaries.append("")
        bart_all_summaries.append("")
        continue

    print(f"\nProcessing transcript at index {idx}...")
    
    # Summary using transcript only.
    try:
        t5_trans = summarize_t5(transcript)
        bart_trans = summarize_bart(transcript)
    except Exception as e:
        print(f"Error processing transcript-only summary at index {idx}: {e}")
        t5_trans, bart_trans = "", ""
    
    # Prepare a combined text with all the fields.
    # Here we concatenate other column values. You may adjust the formatting as needed.
    all_text = (
        f"Transcript: {row['transcript']}. "
        f"Word Count: {row['word_count']}. "
        f"Sentence Count: {row['sentence_count']}. "
        f"Motion Count: {row['motion_count']}. "
        f"Average Word Length: {row['avg_word_len']}. "
        f"Sentiment: {row['sentiment']}."
    )
    
    try:
        t5_all = summarize_t5(all_text)
        bart_all = summarize_bart(all_text)
    except Exception as e:
        print(f"Error processing full summary at index {idx}: {e}")
        t5_all, bart_all = "", ""
    
    # Store the generated summaries.
    t5_transcript_summaries.append(t5_trans)
    bart_transcript_summaries.append(bart_trans)
    t5_all_summaries.append(t5_all)
    bart_all_summaries.append(bart_all)

# Create DataFrames for each set of summaries.
t5_trans_df = pd.DataFrame({"summary": t5_transcript_summaries})
bart_trans_df = pd.DataFrame({"summary": bart_transcript_summaries})
t5_all_df = pd.DataFrame({"summary": t5_all_summaries})
bart_all_df = pd.DataFrame({"summary": bart_all_summaries})

# Save each summary DataFrame to its own gzip-compressed CSV file inside the data folder.
t5_trans_csv = os.path.join(data_folder, "t5_summaries.csv.gz")
bart_trans_csv = os.path.join(data_folder, "bart_summaries.csv.gz")
t5_all_csv = os.path.join(data_folder, "t5_metadata_summaries.csv.gz")
bart_all_csv = os.path.join(data_folder, "bart_metadata_summaries.csv.gz")

t5_trans_df.to_csv(t5_trans_csv, index=False, compression="gzip")
bart_trans_df.to_csv(bart_trans_csv, index=False, compression="gzip")
t5_all_df.to_csv(t5_all_csv, index=False, compression="gzip")
bart_all_df.to_csv(bart_all_csv, index=False, compression="gzip")

print(f"\nT5 transcript summaries saved to {t5_trans_csv}")
print(f"BART transcript summaries saved to {bart_trans_csv}")
print(f"T5 full summaries saved to {t5_all_csv}")
print(f"BART full summaries saved to {bart_all_csv}")

