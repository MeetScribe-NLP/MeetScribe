import pandas as pd
import matplotlib.pyplot as plt
import os

# === Paths ===
current_dir = os.path.dirname(os.path.abspath(__file__))
metrics_dir = os.path.join(current_dir, "summaries_metrics")
output_dir = os.path.join(current_dir, "evaluation_graphs")
os.makedirs(output_dir, exist_ok=True)

# === File paths ===
abstractive_files = {
    "with_metadata": os.path.join(metrics_dir, "bertscore_abstractive_with_metadata.csv"),
    "without_metadata": os.path.join(metrics_dir, "bertscore_abstractive_without_metadata.csv")
}
extractive_files = {
    "with_metadata": os.path.join(metrics_dir, "bertscore_extractive_with_metadata.csv"),
    "without_metadata": os.path.join(metrics_dir, "bertscore_extractive_without_metadata.csv")
}

# === Models to plot ===
models = ['T5', 'BART', 'OpenAI', 'TextRank', 'LexRank', 'BertSUM']
with_meta_scores = []
without_meta_scores = []

# === Extract scores ===
# Abstractive
for label, file_path in abstractive_files.items():
    df = pd.read_csv(file_path)
    scores = [
        df['t5'].mean(),
        df['bart'].mean(),
        df['openai'].mean()
    ]
    if label == "with_metadata":
        with_meta_scores.extend(scores)
    else:
        without_meta_scores.extend(scores)

# Extractive
for label, file_path in extractive_files.items():
    df = pd.read_csv(file_path)
    scores = [
        df['bertscore_textrank'].mean(),
        df['bertscore_lexrank'].mean(),
        df['bertscore_bertsum'].mean()
    ]
    if label == "with_metadata":
        with_meta_scores.extend(scores)
    else:
        without_meta_scores.extend(scores)

# === Plotting ===
plt.figure(figsize=(12, 6))
plt.plot(models, with_meta_scores, marker='o', label='With Metadata')
plt.plot(models, without_meta_scores, marker='s', label='Without Metadata')

# Add ideal BERTScore line for meeting summaries
ideal_bertscore = 0.90
plt.axhline(ideal_bertscore, color='gray', linestyle='--', linewidth=1.5, label='Ideal Score (0.90)')

plt.title("Semantic Similarity Assessment (BERTScore)")
plt.xlabel("Model")
plt.ylabel("Average BERTScore (F1)")
plt.ylim(0.75, 1.0)
plt.legend()
plt.tight_layout()

# No grid for a cleaner look
# plt.grid(False) ← unnecessary since grid is off by default unless enabled

# Save the plot
output_path = os.path.join(output_dir, "semantic_similarity_bertscore.png")
plt.savefig(output_path)
plt.close()

print(f"Line graph with ideal BERTScore saved to: {output_path}")
