import os

label_dir = "labels/all"

for file in os.listdir(label_dir):
    if file.lower().startswith("cardboard") and file.endswith(".txt"):
        path = os.path.join(label_dir, file)
        with open(path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            parts[0] = "1"  # paper
            new_lines.append(" ".join(parts))

        with open(path, "w") as f:
            f.write("\n".join(new_lines))

print("✅ All cardboard labels fixed to PAPER (class 1)")
