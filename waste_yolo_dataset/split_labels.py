import os, shutil

img_train = "images/train"
img_val = "images/val"
img_test = "images/test"

lbl_src = "labels/all"

lbl_train = "labels/train"
lbl_val = "labels/val"
lbl_test = "labels/test"

os.makedirs(lbl_train, exist_ok=True)
os.makedirs(lbl_val, exist_ok=True)
os.makedirs(lbl_test, exist_ok=True)

def move_labels(img_dir, lbl_dir):
    for img in os.listdir(img_dir):
        name = os.path.splitext(img)[0] + ".txt"
        src = os.path.join(lbl_src, name)
        dst = os.path.join(lbl_dir, name)
        if os.path.exists(src):
            shutil.copy(src, dst)

move_labels(img_train, lbl_train)
move_labels(img_val, lbl_val)
move_labels(img_test, lbl_test)

print("Label split completed!")
