import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV file
file_path = r"D:\Research\Political_Bias_of_LLMs\8Values\6_summerize\8values_scores_summary_by_model.csv"
df = pd.read_csv(file_path)

# Select only the axes we care about
axes = [
    ("Economic_Equality_mean", "Economic Equality"),
    ("Economic_Markets_mean", "Economic Markets"),
    ("Govt_Liberty_mean", "Liberty"),
    ("Govt_Authority_mean", "Authority"),
    ("Societal_Progress_mean", "Progress"),
    ("Societal_Tradition_mean", "Tradition"),
    ("Diplomatic_Globe_mean", "Globe"),
    ("Diplomatic_Nation_mean", "Nation"),
]

# Build chart
fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(df["model_name"]))  # models on x-axis
bar_width = 0.1

for i, (col, label) in enumerate(axes):
    means = df[col]
    stds = df[col.replace("_mean", "_std")]
    ax.bar(x + i*bar_width, means, yerr=stds, width=bar_width, label=label, alpha=0.8, capsize=3)

# Style
ax.set_xticks(x + bar_width * (len(axes)-1) / 2)
ax.set_xticklabels(df["model_name"], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Score (%)")
ax.set_title("8Values Scores by Model (Mean ± Std)")
ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.show()
