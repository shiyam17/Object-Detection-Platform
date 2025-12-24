import os
import cv2
import yaml
import torch
from ultralytics import YOLO
from datetime import datetime
import matplotlib.pyplot as plt

# ---------------- PATH SETUP (DYNAMIC) ---------------- #
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_YAML_PATH = os.path.join(BASE_DIR, "dataset", "data.yaml")
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "dataset", "test", "images")

MODEL_PATH = os.path.join(
    BASE_DIR,
    "runs",
    "waste_yolov8_20251222_133043",
    "weights",
    "best.pt"
)

RUNS_DIR = os.path.join(BASE_DIR, "runs", "detect")
# ----------------------------------------------------- #


def test_on_images(conf_threshold=0.25, max_images=None):
    # Device selection
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")

    # Load YOLO model
    model = YOLO(MODEL_PATH)

    # Load class names
    with open(DATA_YAML_PATH, "r") as f:
        class_names = yaml.safe_load(f)["names"]

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(RUNS_DIR, f"predict_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Collect test images
    image_files = [
        f for f in os.listdir(TEST_IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if max_images:
        image_files = image_files[:max_images]

    print(f"🖼️ Running inference on {len(image_files)} images")

    for img_name in image_files:
        img_path = os.path.join(TEST_IMAGES_DIR, img_name)

        # Run inference
        results = model.predict(
            source=img_path,
            conf=conf_threshold,
            device=device,
            verbose=False
        )

        # Read image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Draw detections
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            label = f"{class_names[cls]} {conf:.2f}"
            color = (cls * 70 % 255, cls * 50 % 255, cls * 90 % 255)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img,
                label,
                (x1, max(y1 - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        # Save output image
        save_path = os.path.join(output_dir, img_name)
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    print("\n✅ Inference completed successfully")
    print(f"📂 Results saved to:\n{output_dir}")


if __name__ == "__main__":
    # Run on ALL test images by default
    test_on_images()
