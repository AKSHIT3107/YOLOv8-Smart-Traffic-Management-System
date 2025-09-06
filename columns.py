import pandas as pd

df = pd.read_csv("smart_traffic/emergency_vehicle_model25/results.csv")
print(df.columns.tolist())
