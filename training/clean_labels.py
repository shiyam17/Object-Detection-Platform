import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LABEL_DIRS = [
    os.path.join(BASE_DIR, "dataset", "train", "labels"),
    os.path.join(BASE_DIR, "dataset", "valid", "labels"),
    os.path.join(BASE_DIR, "dataset", "test", "labels"),
]

# 🔁 OLD → NEW CLASS ID MAP (EDIT THIS)
CLASS_ID_MAP = {
    1: 0,
    3: 1,
    5: 2,
    7: 3,
    10: 4,
    12: 5,
    13: 6,
    16: 7,
    17: 8,
    18: 9,
    25: 10,
    27: 11,
    32: 12,
    35: 13,
    36: 14,
}

def clean_labels():
    total_files = 0
    cleaned_files = 0
    removed_boxes = 0

    for label_dir in LABEL_DIRS:
        for file in os.listdir(label_dir):
            if not file.endswith(".txt"):
                continue

            total_files += 1
            file_path = os.path.join(label_dir, file)

            with open(file_path, "r") as f:
                lines = f.readlines()

            new_lines = []

            for line in lines:
                parts = line.strip().split()
                old_id = int(parts[0])

                if old_id in CLASS_ID_MAP:
                    parts[0] = str(CLASS_ID_MAP[old_id])
                    new_lines.append(" ".join(parts) + "\n")
                else:
                    removed_boxes += 1

            if new_lines:
                with open(file_path, "w") as f:
                    f.writelines(new_lines)
                cleaned_files += 1
            else:
                os.remove(file_path)  # remove empty label files

    print("\n✅ LABEL CLEANING COMPLETED")
    print(f"📄 Total label files scanned : {total_files}")
    print(f"🧹 Cleaned label files       : {cleaned_files}")
    print(f"❌ Removed bounding boxes    : {removed_boxes}")

if __name__ == "__main__":
    clean_labels()
