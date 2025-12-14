# coco_to_yolo.py
import json, os
from PIL import Image
from tqdm import tqdm
import argparse

def coco_to_yolo(coco_json, images_dir, out_labels_dir, classes_list=None):
    with open(coco_json, 'r') as f:
        coco = json.load(f)

    # id -> filename
    imgs = {im['id']: im for im in coco['images']}
    # build class list if not provided (keep consistent ordering)
    if classes_list is None:
        cat_ids = sorted({cat['id'] for cat in coco['categories']})
        classes = {cat['id']: i for i,cat in enumerate(sorted(coco['categories'], key=lambda x: x['id']))}
    else:
        classes = {cat_id: classes_list.index(cat_name) for cat_id,cat_name in enumerate(classes_list)} # if you want custom mapping

    os.makedirs(out_labels_dir, exist_ok=True)

    for ann in tqdm(coco['annotations']):
        img = imgs[ann['image_id']]
        fn = img['file_name']
        H, W = img.get('height'), img.get('width')
        if H is None or W is None:
            p = os.path.join(images_dir, fn)
            with Image.open(p) as im:
                W, H = im.size

        x, y, w, h = ann['bbox']  # COCO bbox: [x_min, y_min, width, height]
        x_center = x + w/2
        y_center = y + h/2

        x_center /= W
        y_center /= H
        w /= W
        h /= H

        class_id = ann['category_id'] - min(coco['categories'], key=lambda x:x['id'])['id']  # adjust if needed

        label_path = os.path.splitext(os.path.join(out_labels_dir, fn))[0] + '.txt'
        with open(label_path, 'a') as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco', required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    coco_to_yolo(args.coco, args.images, args.out)
