# remove_corrupted.py
from PIL import Image
import os, sys
from tqdm import tqdm

def is_image_ok(path):
    try:
        with Image.open(path) as img:
            img.verify()   # verifies file integrity
        return True
    except Exception as e:
        return False

def scan_and_remove(folder, remove=False):
    exts = ('.jpg','.jpeg','.png','.bmp')
    files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(folder) for f in filenames if f.lower().endswith(exts)]
    bad = []
    for f in tqdm(files):
        if not is_image_ok(f):
            bad.append(f)
            if remove:
                os.remove(f)
    return bad

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv)>1 else "waste_yolo_dataset/raw"
    bad = scan_and_remove(folder, remove=False)
    print(f"Found {len(bad)} corrupted images.")
    for b in bad[:20]:
        print(b)
