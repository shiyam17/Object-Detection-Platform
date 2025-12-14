import cv2
import os
from tqdm import tqdm

def preprocess_image(img):
    img = cv2.resize(img, (640, 640))
    img = cv2.GaussianBlur(img, (3, 3), 0)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    lab = cv2.merge((cl, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return img


def process_split(split):
    in_dir = f"images/{split}"
    out_dir = f"images/{split}_preprocessed"
    os.makedirs(out_dir, exist_ok=True)

    for img_name in tqdm(os.listdir(in_dir), desc=f"Processing {split}"):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(in_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            processed = preprocess_image(img)
            cv2.imwrite(os.path.join(out_dir, img_name), processed)


# Run for all splits
for split in ["train", "val", "test"]:
    process_split(split)
