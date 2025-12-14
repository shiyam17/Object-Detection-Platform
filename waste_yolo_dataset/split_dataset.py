import os, random, shutil

SOURCE = "images/all"
DEST = "images"

train_dir = f"{DEST}/train"
val_dir = f"{DEST}/val"
test_dir = f"{DEST}/test"

# create folders
for d in [train_dir, val_dir, test_dir]:
    os.makedirs(d, exist_ok=True)

images = [f for f in os.listdir(SOURCE) if f.lower().endswith(('.jpg','.png','.jpeg'))]
random.shuffle(images)

train_split = 0.7
val_split = 0.2

train_end = int(len(images) * train_split)
val_end = train_end + int(len(images) * val_split)

for i, img in enumerate(images):
    src = os.path.join(SOURCE, img)
    if i < train_end:
        dst = os.path.join(train_dir, img)
    elif i < val_end:
        dst = os.path.join(val_dir, img)
    else:
        dst = os.path.join(test_dir, img)

    shutil.copy(src, dst)

print("Dataset split complete!")
