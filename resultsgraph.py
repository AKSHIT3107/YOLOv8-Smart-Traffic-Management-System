import pandas as pd
import matplotlib.pyplot as plt
import os

# 📂 Path to results.csv
csv_path = "smart_traffic/emergency_vehicle_model25/results.csv"
output_dir = "smart_traffic/emergency_vehicle_model25/plots"
os.makedirs(output_dir, exist_ok=True)

# 📊 Load data
df = pd.read_csv(csv_path)
epochs = df["epoch"]

# 📈 Plot mAP
plt.figure(figsize=(10, 6))
plt.plot(epochs, df["metrics/mAP50(B)"], label="mAP@0.5", color="blue")
plt.plot(epochs, df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95", color="purple")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("YOLOv8 mAP Scores Over Epochs")
plt.legend()
plt.grid(True)
plt.savefig(f"{output_dir}/map_only.png")
plt.close()

print(f"✅ mAP plot saved at: {output_dir}/map_only.png")
