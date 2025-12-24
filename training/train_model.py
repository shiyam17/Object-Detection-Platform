import torch
from ultralytics import YOLO

def train():
    # 🔒 Force GPU (no fallback)
    assert torch.cuda.is_available(), "❌ CUDA not available"
    device = "cuda:0"
    print(f"🚀 Training on GPU: {torch.cuda.get_device_name(0)}")

    # Use medium model for better precision
    model = YOLO("yolov8m.pt")

    model.train(
        data="dataset/data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,          # Safe for RTX 3050
        device=device,
        workers=4,

        # 🎯 Precision tuning
        iou=0.75,
        cls=0.8,
        box=8.0,

        # Reduce noisy augmentation
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,

        patience=20,
        plots=True,
        amp=True           # Mixed precision (GPU safe)
    )

if __name__ == "__main__":
    train()
