# verify_labels.py
import os
from PIL import Image

def verify(image_folder, label_folder, classes_file=None):
    errors = []
    if classes_file:
        with open(classes_file) as f:
            classes = [x.strip() for x in f if x.strip()]
    else:
        classes = None
    for img_name in os.listdir(image_folder):
        if not img_name.lower().endswith(('.jpg','.png','.jpeg')): continue
        img_path = os.path.join(image_folder, img_name)
        label_path = os.path.splitext(os.path.join(label_folder, img_name))[0]+'.txt'
        if not os.path.exists(label_path):
            errors.append((img_name, "no_label"))
            continue
        with Image.open(img_path) as im:
            w,h = im.size
        with open(label_path) as f:
            for i,line in enumerate(f):
                parts = line.strip().split()
                if len(parts)!=5:
                    errors.append((img_name, f"bad_format_line_{i+1}"))
                    continue
                cls, xc, yc, nw, nh = parts
                try:
                    cls = int(cls)
                    xc, yc, nw, nh = map(float, (xc,yc,nw,nh))
                except:
                    errors.append((img_name,"invalid_numbers"))
                    continue
                if not (0<=xc<=1 and 0<=yc<=1 and 0<=nw<=1 and 0<=nh<=1):
                    errors.append((img_name,"coords_out_of_range"))
                if classes and (cls<0 or cls>=len(classes)):
                    errors.append((img_name,"class_id_out_of_range"))
    return errors

if __name__ == "__main__":
    errs = verify("images/train","labels/train","classes.txt")

    print("Errors found:", len(errs))
    for e in errs[:30]:
        print(e)
