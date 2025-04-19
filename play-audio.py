import pandas as pd
from IPython.display import Audio

# Load the CSV file
df = pd.read_csv('meetingBank_styled.csv') # // Final dataset

# Function to play audio from a specified row
def play_audio_from_row(row_num):
    try:
        audio_path = df.loc[row_num, 'audio_path']
        return Audio(audio_path, autoplay=True)
    except Exception as e:
        print(f"Error: {e}")

# Example usage: Play audio from row 0
play_audio_from_row(0)