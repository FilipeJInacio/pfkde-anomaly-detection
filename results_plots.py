import os
import pandas as pd
import json

with open("config.json", "r") as f:
    config = json.load(f)

k_values = config["k_values"]

results = {}

# newest execution folder
date_folders = [f for f in os.listdir("results") if os.path.isdir(os.path.join("results", f))]
newest_date_folder = max(date_folders)
base_path = os.path.join("results", newest_date_folder)

# get results.csv into a pd
results_csv_path = os.path.join(base_path, "results.csv")
if os.path.exists(results_csv_path):
    results = pd.read_csv(results_csv_path)

# do a plot where x axis is results['dataset'] and y axis is results['f1_score']
# the points should only connect if they have the same results['algorithm']

import matplotlib.pyplot as plt

metrics = [('F1Score_PercentileThresholding(percentile=99.47021931362238)', 'F1 Score'), ('Precision_PercentileThresholding(percentile=99.47021931362238)', 'Precision'),
           ('Recall_PercentileThresholding(percentile=99.47021931362238)', 'Recall'), ('ROC_AUC', 'ROC AUC')]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for ax, (metric_col, metric_name) in zip(axes, metrics):

    for algorithm in results['algorithm'].unique():
        subset = results[results['algorithm'] == algorithm]

        ax.plot(k_values, subset[metric_col], marker='o', label=algorithm)

    ax.set_xlabel('k')
    #ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} by Dataset and Algorithm')
    ax.tick_params(axis='x')
    ax.grid(True)

# Show legend only once to avoid clutter
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=min(len(labels), 6))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
