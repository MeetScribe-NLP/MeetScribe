import os
import re
import pandas as pd
import gzip

# Define only the extra filler words (do not include full stopword lists)
filler_words = {
    "um", "uh", "like", "you know", "i mean", "well", "so", "basically", "right",
    "actually", "literally", "honestly", "seriously", "okay", "alright", "anyway",
    "yeah", "kind of", "sort of", "just", "totally", "really"
}

# Define procedural phrases (common in meetings)
procedural_phrases = {
    "good morning everyone", "welcome to the meeting", "let's get started",
    "moving on", "next slide please", "thank you for your time",
    "can you hear me", "does that make sense", "let’s circle back",
    "just a quick note", "before we begin", "quick question", "going forward"
}

# Get script's directory and parent directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

# Define folder for data input and output
data_folder = os.path.join(parent_dir, "data")

# Define the input zipped CSV file path
data_file_path = os.path.join(data_folder, "meetingBank.csv.gz")

# Check if the data file exists before loading
if not os.path.exists(data_file_path):
    raise FileNotFoundError(f"Data file not found at {data_file_path}")

# Unzip and load the CSV file into a DataFrame
with gzip.open(data_file_path, 'rt', encoding='utf-8') as f_in:
    df_combined = pd.read_csv(f_in)

def clean_text(text):
    if not isinstance(text, str):
        return text

    print("\nBefore Cleaning:", text)
    cleaned_text = text

    # 1. Remove procedural phrases
    for phrase in procedural_phrases:
        pattern = rf'\b{re.escape(phrase)}\b'
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)

    # 2. Remove filler words (using exact word boundaries)
    for phrase in filler_words:
        pattern = rf'\b{re.escape(phrase)}\b'
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)

    # 3. Clean up extra spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    # 4. Remove stray punctuation (e.g. leftover commas or semicolons)
    cleaned_text = re.sub(r'^[,;\s]+', '', cleaned_text)
    cleaned_text = re.sub(r'\s*[,;]+\s*', ' ', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    print("After Cleaning:", cleaned_text)
    return cleaned_text

# Apply text cleaning to the 'transcript' column if it exists
if "transcript" in df_combined.columns:
    df_combined["transcript"] = df_combined["transcript"].apply(clean_text)
else:
    print("Warning: 'transcript' column not found in the dataset!")

# Define the output zipped CSV file path for the cleaned data
cleaned_file_path = os.path.join(data_folder, "meetingBank_cleaned.csv.gz")

# Save the cleaned DataFrame as a gzipped CSV file
df_combined.to_csv(cleaned_file_path, index=False, compression="gzip")

print(f"\nCleaned transcript saved at {cleaned_file_path}")
