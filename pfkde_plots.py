from pfkde.model import PFKDE
import pandas as pd
import numpy as np
import json
with open("config.json", "r") as f:
    config = json.load(f)

path_to_plots = config["path_to_plots"]

# from CSV import dataset
df = pd.read_csv("timeeval-datasets/multivariate/Bugsat/Bugsat.test.csv")

# format as in the time eval
# output (n, 4): timestamp, value, phase, period_index + (n, 1): label
data = df[["timestamp", "nice_battery_mv", "phase", "period_index"]].values
labels = df["is_anomaly"].values

# How many anomalies?
print(f"Total number of points: {len(labels)}, {len(data)}")
print(f"Number of anomalies: {labels.sum()} out of {len(labels)}")
print(f"Contamination: {labels.sum() / len(labels):.8f}")

model = PFKDE(
    n_bins=5782,
    omission_threshold=10**-9,
    n_minimum_points=15,
    aggregation_window_size=15,
    memory_size=300,
    bandwidth_function=1,
    weight_function=1,

    threshold_type=1,
    contamination=sum(labels) / len(labels),

    plot=True,
    labels=labels,
    y_bottom=10500,
    y_upper=13000,
    precision=200,
    fig_path=path_to_plots,
    frame_n=50,
)
model.fit(data)
