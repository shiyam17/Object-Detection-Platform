import os
import yaml

# ---------------- CONFIG ---------------- #
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DATA_YAML_PATH = os.path.join(DATASET_DIR, "data.yaml")

SPLITS = ["train", "valid", "test"]

# ---------------------------------------- #

def verify_dataset_structure():
    print("\n🔍 Verifying dataset structure...\n")

    # 1️⃣ Check dataset folders
    for split in SPLITS:
        images_path = os.path.join(DATASET_DIR, split, "images")
        labels_path = os.path.join(DATASET_DIR, split, "labels")

        if not os.path.exists(images_path):
            raise FileNotFoundError(f"❌ Missing folder: {images_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"❌ Missing folder: {labels_path}")

    print("✅ Folder structure OK")

    # 2️⃣ Load or create data.yaml
    if not os.path.exists(DATA_YAML_PATH):
        raise FileNotFoundError("❌ data.yaml not found")

    with open(DATA_YAML_PATH, "r") as f:
        data = yaml.safe_load(f)

    class_names = data.get("names", [])
    nc = data.get("nc", 0)

    if len(class_names) != nc:
        raise ValueError("❌ nc does not match number of class names")

    print(f"✅ Classes verified: {nc} classes")

    # 3️⃣ Count images & labels
    for split in SPLITS:
        img_dir = os.path.join(DATASET_DIR, split, "images")
        lbl_dir = os.path.join(DATASET_DIR, split, "labels")

        images = [f for f in os.listdir(img_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        labels = [f for f in os.listdir(lbl_dir) if f.endswith(".txt")]

        print(f"\n📁 {split.upper()}")
        print(f"   Images: {len(images)}")
        print(f"   Labels: {len(labels)}")

        if len(images) == 0 or len(labels) == 0:
            raise ValueError(f"❌ {split} split is empty")

        if len(images) != len(labels):
            print(f"⚠️ Warning: Image-label count mismatch in {split}")

    print("\n🎉 Dataset verification SUCCESSFUL")
    return True


if __name__ == "__main__":
    verify_dataset_structure()
