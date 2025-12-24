import pandas as pd

CSV_PATH = r"C:\wastededectionproject\runs\waste_yolov8_20251222_133043\results.csv"

df = pd.read_csv(CSV_PATH)

last = df.iloc[-1]

print("\n✅ LAST EPOCH METRICS")
print("-" * 45)
print("Epoch       :", int(last["epoch"]))
print("Precision   :", round(last["metrics/precision(B)"], 3))
print("Recall      :", round(last["metrics/recall(B)"], 3))
print("mAP@50      :", round(last["metrics/mAP50(B)"], 3))
print("mAP@50-95   :", round(last["metrics/mAP50-95(B)"], 3))
print("-" * 45)
