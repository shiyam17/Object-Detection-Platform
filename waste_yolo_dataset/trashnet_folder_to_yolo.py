# trashnet_folder_to_yolo.py
import cv2, os
import numpy as np
from tqdm import tqdm

def largest_bbox_from_mask(img):
    # Convert to gray then threshold to isolate object
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,th = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    if w*h < 50:  # ignore tiny
        return None
    return x,y,w,h

def process(root_dir, out_label_dir, classes_map=None, img_out_dir=None, resize=None):
    os.makedirs(out_label_dir, exist_ok=True)
    if img_out_dir:
        os.makedirs(img_out_dir, exist_ok=True)

    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,d))])
    cls_to_id = {c:i for i,c in enumerate(classes)} if classes_map is None else classes_map

    for cls in classes:
        folder = os.path.join(root_dir, cls)
        for fname in tqdm(os.listdir(folder)):
            if not fname.lower().endswith(('.jpg','.png','.jpeg')): continue
            p = os.path.join(folder, fname)
            img = cv2.imread(p)
            if img is None: continue
            h, w = img.shape[:2]
            bbox = largest_bbox_from_mask(img)
            if bbox is None:
                # fallback: use full image as box
                x, y, bw, bh = 0, 0, w, h
            else:
                x,y,bw,bh = bbox

            # normalize
            x_center = (x + bw/2) / w
            y_center = (y + bh/2) / h
            nw = bw / w
            nh = bh / h

            class_id = cls_to_id[cls]
            label_path = os.path.join(out_label_dir, os.path.splitext(fname)[0] + '.txt')
            with open(label_path, 'w') as f:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {nw:.6f} {nh:.6f}\n")

            # optionally copy/rescale image to unified images folder
            if img_out_dir:
                if resize:
                    img2 = cv2.resize(img, resize)
                else:
                    img2 = img
                cv2.imwrite(os.path.join(img_out_dir, fname), img2)

if __name__ == "__main__":
    process(
        root_dir="raw/trashnet",
        out_label_dir="labels/all",
        img_out_dir="images/all",
        resize=(640, 640)
    )

