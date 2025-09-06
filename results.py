import pandas as pd
import matplotlib.pyplot as plt
import os

# ✅ Your CSV path
csv_path = "smart_traffic/emergency_vehicle_model25/results.csv"
output_dir = "smart_traffic/emergency_vehicle_model25/plots"
os.makedirs(output_dir, exist_ok=True)

# 📈 Load results
df = pd.read_csv(csv_path)
epochs = df["epoch"]

# 📊 1. Precision & Recall
plt.figure(figsize=(10, 6))
plt.plot(epochs, df["metrics/precision(B)"], label="Precision", color="green")
plt.plot(epochs, df["metrics/recall(B)"], label="Recall", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Precision & Recall per Epoch")
plt.legend()
plt.grid(True)
plt.savefig(f"{output_dir}/precision_recall.png")
plt.close()

# 📊 2. mAP@0.5 and mAP@0.5:0.95
plt.figure(figsize=(10, 6))
plt.plot(epochs, df["metrics/mAP50(B)"], label="mAP@0.5", color="blue")
plt.plot(epochs, df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95", color="purple")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("mAP Scores per Epoch")
plt.legend()
plt.grid(True)
plt.savefig(f"{output_dir}/map_scores.png")
plt.close()

# 📊 3. Loss Curves
plt.figure(figsize=(10, 6))
plt.plot(epochs, df["train/box_loss"], label="Box Loss", color="red")
plt.plot(epochs, df["train/cls_loss"], label="Class Loss", color="blue")
plt.plot(epochs, df["train/dfl_loss"], label="DFL Loss", color="green")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Losses per Epoch")
plt.legend()
plt.grid(True)
plt.savefig(f"{output_dir}/loss_curves.png")
plt.close()

print(f"✅ Graphs saved to: {output_dir}")
