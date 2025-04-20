import pandas as pd
from gtts import gTTS
from IPython.display import Audio
import os


# Load the CSV file
df = pd.read_csv('summarized-data/openai_transcript_summaries.csv.gz', compression='gzip') #Any summary file can be added here

# Function to generate and play audio from transcript
def play_audio_from_transcript(row_num):
    try:
        text = df.loc[row_num, 'summary']
        tts = gTTS(text)
        
        # Save as temporary MP3
        temp_audio_path = "temp_audio.mp3"
        tts.save(temp_audio_path)

        # Play audio
        return Audio(filename=temp_audio_path, autoplay=True)
    
    except Exception as e:
        print(f"Error: {e}")

# Example: play row 0
play_audio_from_transcript(1) # Any row can be selected and mp3 audio can be generated. Output file name is temp_audio.mp3