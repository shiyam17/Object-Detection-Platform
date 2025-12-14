from ultralytics import YOLO

if __name__ == "__main__":
    # Load model
    model = YOLO("yolov8s.pt")

    # Train on GPU
    model.train(
        data="waste.yaml",
        epochs=30,
        imgsz=640,
        batch=16,
        device=0
        )


    # Validate
    model.val()

    # Predict
    model.predict(
        source="images/test_preprocessed",
        save=True,
        device=0
    )
