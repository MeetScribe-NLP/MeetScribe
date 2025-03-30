from datasets import load_dataset
import pandas as pd
import os
import gzip
import shutil

# Load the dataset
meetingbank = load_dataset("huuuyeah/meetingbank")

# Convert each split to a Pandas DataFrame
df_train = meetingbank["train"].to_pandas()
df_test = meetingbank["test"].to_pandas()
df_validation = meetingbank["validation"].to_pandas()

# Merge all splits into a single DataFrame
df_combined = pd.concat([df_train, df_test, df_validation], ignore_index=True)

# Remove the 'summary', 'uid', and 'id' columns
df_combined = df_combined.drop(columns=["summary", "uid", "id"])

# Define folder and file path
current_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the script's folder
folder_path = os.path.join(current_dir, "data")
os.makedirs(folder_path, exist_ok=True)
file_path = os.path.join(folder_path, "meetingBank.csv")

# Save the DataFrame as a CSV file (overwrite if it exists)
df_combined.to_csv(file_path, index=False)
print(f"Data successfully saved at {file_path}")

# Compress the CSV file using gzip
compressed_file_path = os.path.join(folder_path, "meetingBank.csv.gz")
with open(file_path, 'rb') as f_in:
    with gzip.open(compressed_file_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

print(f"Compressed CSV file saved at {compressed_file_path}")
