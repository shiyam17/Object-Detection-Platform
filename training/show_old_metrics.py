import pandas as pd

CSV_PATH = r"runs/waste_yolov8_20251222_133043/results.csv"

df = pd.read_csv(CSV_PATH)

# Find epoch with best precision
best_row = df.loc[df["metrics/precision(B)"].idxmax()]

print("\n✅ OLD MODEL BEST METRICS (FROM TRAINING LOGS)")
print("-" * 50)
print("Epoch       :", int(best_row["epoch"]))
print("Precision   :", round(best_row["metrics/precision(B)"], 3))
print("Recall      :", round(best_row["metrics/recall(B)"], 3))
print("mAP@50      :", round(best_row["metrics/mAP50(B)"], 3))
print("mAP@50-95   :", round(best_row["metrics/mAP50-95(B)"], 3))
print("-" * 50)
