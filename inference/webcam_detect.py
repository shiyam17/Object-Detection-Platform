import os
import cv2
import yaml
import torch
from ultralytics import YOLO

# ---------------- PATH SETUP (DYNAMIC) ---------------- #
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "runs",
    "waste_yolov8_20251222_133043",
    "weights",
    "best.pt"
)

DATA_YAML_PATH = os.path.join(BASE_DIR, "dataset", "data.yaml")
# ----------------------------------------------------- #


def webcam_detection(conf_threshold=0.6):
    # Device selection
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")

    # Load model
    model = YOLO(MODEL_PATH)

    # Load class names
    with open(DATA_YAML_PATH, "r") as f:
        class_names = yaml.safe_load(f)["names"]

    # Open webcam (0 = default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot access webcam")
        return

    print("🎥 Webcam started — press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model.predict(
            source=frame,
            conf=conf_threshold,
            device=device,
            verbose=False
        )

        # Draw detections
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            label = f"{class_names[cls]} {conf:.2f}"
            color = (cls * 60 % 255, cls * 120 % 255, cls * 30 % 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        # Show output
        cv2.imshow("Waste Detection - Webcam", frame)

        # Exit key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Webcam closed")


if __name__ == "__main__":
    webcam_detection()
