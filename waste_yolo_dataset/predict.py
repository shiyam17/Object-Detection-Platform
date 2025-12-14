from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train9/weights/best.pt")

# Run prediction on test images
model.predict(
    source="images/test_preprocessed",
    imgsz=640,
    conf=0.25,
    save=True
)

print("✅ Prediction completed. Check runs/detect/predict")
