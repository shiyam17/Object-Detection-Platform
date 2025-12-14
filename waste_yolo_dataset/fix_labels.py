# fix_labels.py
import os

label_dirs = [
    "labels/train_preprocessed",
    "labels/val_preprocessed",
    "labels/test_preprocessed"
]

for d in label_dirs:
    for file in os.listdir(d):
        path = os.path.join(d, file)
        with open(path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            cls = int(parts[0])
            if cls > 4:
                cls = 4   # map invalid class to trash
            parts[0] = str(cls)
            new_lines.append(" ".join(parts))

        with open(path, "w") as f:
            f.write("\n".join(new_lines))
