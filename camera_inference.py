import cv2
import torch
from ultralytics import YOLO

# ---------------- PATHS ---------------- #
MODEL_PATH = r"C:\wastededectionproject\runs\waste_yolov8_20251222_133043\weights\best.pt"
# --------------------------------------- #

def run_camera():
    # Device selection
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Running on device: {device}")

    # Load model
    model = YOLO(MODEL_PATH)

    # Open webcam (0 = default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera not accessible")
        return

    print("📸 Camera started. Press 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO inference
        results = model.predict(
            source=frame,
            conf=0.5,          # Higher confidence → higher precision
            device=device,
            verbose=False
        )

        # Draw detections
        annotated_frame = results[0].plot()

        # Show output
        cv2.imshow("YOLO Waste Detection - Live", annotated_frame)

        # Exit on Q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Camera closed")

if __name__ == "__main__":
    run_camera()
