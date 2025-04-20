import pandas as pd
import matplotlib.pyplot as plt
import os

# Set current directory and output folder
base_dir = "."
output_dir = os.path.join(base_dir, "evaluation_graphs")
os.makedirs(output_dir, exist_ok=True)

# All model names
model_names = ['Bart', 'T5', 'OpenAI', 'TextRank', 'LexRank', 'BertSUM']

# All metrics to plot
metrics = ['fkgl', 'compression_ratio', 'avg_quality_score']

# Custom graph titles
titles = {
    'fkgl': 'Readability Metrics (FKGL)',
    'compression_ratio': 'Compression Ratio Analysis',
    'avg_quality_score': 'Summarization Quality (GPT)'
}

# Ideal values for each metric
ideal_values = {
    'fkgl': 8.0,
    'compression_ratio': 0.20,
    'avg_quality_score': 5
}

# File mapping
files = {}
for model in model_names:
    files[f"{model}_metadata"] = os.path.join(base_dir, "summaries_metrics", f"{model}_metadata_summaries_metrics.csv")
    files[f"{model}_transcript"] = os.path.join(base_dir, "summaries_metrics", f"{model}_transcript_summaries_metrics.csv")

# Initialize data dictionary
data = {metric: {'with_metadata': [], 'without_metadata': []} for metric in metrics}

# Extract and calculate average metrics
for model in model_names:
    meta_path = files[f"{model}_metadata"]
    transcript_path = files[f"{model}_transcript"]

    df_meta = pd.read_csv(meta_path)
    df_transcript = pd.read_csv(transcript_path)

    data['fkgl']['with_metadata'].append(df_meta['fkgl'].mean())
    data['fkgl']['without_metadata'].append(df_transcript['fkgl'].mean())

    data['compression_ratio']['with_metadata'].append(df_meta['compression_ratio'].mean())
    data['compression_ratio']['without_metadata'].append(df_transcript['compression_ratio'].mean())

    avg_meta_quality = df_meta[['coherence', 'coverage', 'fluency']].mean(axis=1).mean()
    avg_transcript_quality = df_transcript[['coherence', 'coverage', 'fluency']].mean(axis=1).mean()

    data['avg_quality_score']['with_metadata'].append(avg_meta_quality)
    data['avg_quality_score']['without_metadata'].append(avg_transcript_quality)

# Plotting and saving graphs
for metric in metrics:
    plt.figure(figsize=(10, 6))
    plt.plot(model_names, data[metric]['with_metadata'], marker='o', label='With Metadata')
    plt.plot(model_names, data[metric]['without_metadata'], marker='s', label='Without Metadata')

    # Add ideal line
    ideal_val = ideal_values[metric]
    plt.axhline(ideal_val, color='gray', linestyle='--', linewidth=1.5, label=f'Ideal: {ideal_val}')

    plt.title(titles[metric])
    plt.xlabel("Model")
    y_labels = {
        'fkgl': 'Average FKGL',
        'compression_ratio': 'Average Compression Ratio',
        'avg_quality_score': 'Average Quality Score'
    }
    plt.ylabel(y_labels[metric])
    plt.legend()
    plt.tight_layout()

    # Save to output folder
    filename = titles[metric].replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_") + ".png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()

print(f"All graphs with ideal scores saved to: {output_dir}")
