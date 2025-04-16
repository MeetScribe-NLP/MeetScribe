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

# Assume that the transcripts are in a column named 'transcript'.
transcript_column = "transcript"
if transcript_column not in df.columns:
    raise ValueError(f"Column '{transcript_column}' not found in the CSV file.")

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
t5_summaries = []
bart_summaries = []

# Iterate over every transcript in the dataframe.
for idx, row in df.iterrows():
    transcript = row[transcript_column]
    if not isinstance(transcript, str) or not transcript.strip():
        # If transcript is empty or not valid, store an empty summary.
        t5_summaries.append("")
        bart_summaries.append("")
        continue

    print(f"\nProcessing transcript at index {idx}...")
    # Generate summaries using both models.
    try:
        t5_sum = summarize_t5(transcript)
        bart_sum = summarize_bart(transcript)
    except Exception as e:
        print(f"Error processing index {idx}: {e}")
        t5_sum = ""
        bart_sum = ""

    t5_summaries.append(t5_sum)
    bart_summaries.append(bart_sum)

# Create separate DataFrames for T5 and BART summaries.
t5_df = pd.DataFrame({"summary": t5_summaries})
bart_df = pd.DataFrame({"summary": bart_summaries})

# Save each summary DataFrame to its own gzip-compressed CSV file inside the data folder.
t5_csv_filename = os.path.join(data_folder, "t5_summaries.csv.gz")
bart_csv_filename = os.path.join(data_folder, "bart_summaries.csv.gz")
t5_df.to_csv(t5_csv_filename, index=False, compression="gzip")
bart_df.to_csv(bart_csv_filename, index=False, compression="gzip")
print(f"\nT5 summaries saved to {t5_csv_filename}")
print(f"BART summaries saved to {bart_csv_filename}")

# Zip the two gzip-compressed CSV files in a single zip archive inside the data folder.
zip_filename = os.path.join(data_folder, "meetingbank_abstractive_summaries.zip")
with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(t5_csv_filename, arcname=os.path.basename(t5_csv_filename))
    zipf.write(bart_csv_filename, arcname=os.path.basename(bart_csv_filename))
print(f"Zipped summaries saved to {zip_filename}")
