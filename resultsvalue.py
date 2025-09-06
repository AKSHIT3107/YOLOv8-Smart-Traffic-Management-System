import pandas as pd

# Load the results.csv file
df = pd.read_csv("smart_traffic/emergency_vehicle_model25/results.csv")

# Define the metrics to evaluate
metrics = {
    "Precision": "metrics/precision(B)",
    "Recall": "metrics/recall(B)",
    "mAP@0.5": "metrics/mAP50(B)",
    "mAP@0.5:0.95": "metrics/mAP50-95(B)"
}

# Print the best values and corresponding epochs
print("📈 Best Metrics Across Training:\n")
for name, column in metrics.items():
    best_epoch = df[column].idxmax()
    best_value = df[column].max()
    print(f"🔹 {name:<12}: {best_value:.3f} at epoch {int(df.loc[best_epoch, 'epoch'])}")
