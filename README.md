# 🚀 Object Detection Platform (YOLOv8)

A scalable, end-to-end **object detection platform** built using **YOLOv8**, designed for both real-time and batch inference across diverse visual datasets. This project demonstrates **production-ready computer vision workflows**, including dataset preprocessing, model training, evaluation, and deployment-ready inference.

---

## 📌 Overview

The **Object Detection Platform** is a modular deep learning system that enables accurate detection and classification of objects from images using state-of-the-art YOLO architectures. The system is designed with **enterprise deployment considerations**, emphasizing scalability, performance, and maintainability.

### Key Objectives
- Build a robust object detection pipeline  
- Support batch and real-time inference  
- Ensure clean dataset handling and reproducible training  
- Generate deployment-ready trained models  

---

## ✨ Key Features

- 🔍 YOLOv8-based object detection  
- ⚡ GPU-accelerated training and inference  
- 🧠 End-to-end ML pipeline (preprocessing → training → validation → prediction)  
- 📊 Evaluation using Precision, Recall, and mAP metrics  
- 🧩 Modular project structure for easy extension  
- 🖼️ Visualized predictions with bounding boxes  
- 🚀 Deployment-ready trained weights  

---
## 🏗️ Pipeline Flow 
Raw Images
↓
Dataset Preprocessing
↓
YOLO Label Generation
↓
Model Training (YOLOv8)
↓
Validation & Evaluation
↓
Inference & Visualization


## 📂 Project Structure

object-detection-platform/
│
├── images/
│ ├── train_preprocessed/
│ ├── val_preprocessed/
│ └── test_preprocessed/
│
├── labels/
│ ├── train_preprocessed/
│ ├── val_preprocessed/
│ └── test_preprocessed/
│
├── runs/
│ └── detect/
│ ├── train/
│ └── predict/
│
├── train_yolo.py
├── predict.py
├── waste.yaml
├── requirements.txt
└── README.md

---

## 🧠 Technology Stack

- **Model:** YOLOv8 (Ultralytics)  
- **Framework:** PyTorch  
- **Language:** Python 3.11  
- **Acceleration:** NVIDIA CUDA (GPU)  
- **Libraries:** OpenCV, NumPy, Ultralytics  

---

## 📊 Training Configuration

- **Input Resolution:** 640 × 640  
- **Optimizer:** AdamW (auto-selected)  
- **Epochs:** 30–50  
- **Batch Size:** GPU-optimized  
- **Evaluation Metrics:**  
  - Precision  
  - Recall  
  - mAP@50  
  - mAP@50–95  

---

## ▶️ How to Run

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt

python train_yolo.py

python predict.py --conf 0.5

runs/detect/predict/

