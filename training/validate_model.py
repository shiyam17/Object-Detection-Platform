import os
import torch
from ultralytics import YOLO

# ---------------- PATH SETUP ---------------- #
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_YAML = os.path.join(BASE_DIR, "dataset", "data.yaml")

MODEL_PATH = os.path.join(
    BASE_DIR,
    "runs",
    "waste_yolov8_20251222_133043",
    "weights",
    "best.pt"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "runs", "final_validation")
# ------------------------------------------- #


def validate_model(model):
    print(f"\n🚀 Running validation on device: {'cuda:0' if torch.cuda.is_available() else 'cpu'}")

    metrics = model.val(
        data=DATA_YAML,
        split="val",
        device=0 if torch.cuda.is_available() else "cpu",
        project=OUTPUT_DIR,
        name="val_results",
        plots=True
    )

    # Metrics (Ultralytics v8)
    precision = metrics.box.mp
    recall = metrics.box.mr
    map50 = metrics.box.map50
    map5095 = metrics.box.map

    f1 = (2 * precision * recall) / (precision + recall + 1e-6)

    print("\n📊 FINAL VALIDATION METRICS")
    print("-" * 50)
    print(f"Precision (mP)  : {precision:.3f}")
    print(f"Recall (mR)     : {recall:.3f}")
    print(f"F1-score        : {f1:.3f}")
    print(f"mAP@50          : {map50:.3f}")
    print(f"mAP@50-95       : {map5095:.3f}")
    print("-" * 50)

    print("\n📁 Validation outputs saved to:")
    print(OUTPUT_DIR)

    return metrics


# ---------------- RUN DIRECTLY ---------------- #
if __name__ == "__main__":
    print(f"\n📦 Loading model:\n{MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    validate_model(model)
