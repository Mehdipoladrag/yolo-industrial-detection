import os
import shutil
import random
from pathlib import Path

ROOT = Path("data/mvtec")
OUT = Path("data/mvtec_yolo")

CLASSES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper"
]

# Create YOLO folder structure
for split in ["train", "val"]:
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

VAL_SPLIT = 0.15


def write_yolo_label(path, w, h, cls_id):
    """Writes YOLO label for full-image bounding box."""
    xc, yc = 0.5, 0.5
    ww, hh = 1.0, 1.0
    with open(path, "w") as f:
        f.write(f"{cls_id} {xc} {yc} {ww} {hh}\n")


def process_class(obj_name, obj_index):
    obj_path = ROOT / obj_name

    good_train = list((obj_path / "train" / "good").glob("*"))
    good_test = list((obj_path / "test" / "good").glob("*"))

    # All bad categories
    bad_folders = [f for f in (obj_path / "test").iterdir() if f.name != "good"]
    bad_images = []
    for folder in bad_folders:
        bad_images += list(folder.glob("*"))

    cls_good = obj_index * 2
    cls_bad = obj_index * 2 + 1

    # Merge all good images
    all_good = good_train + good_test
    random.shuffle(all_good)

    split_idx = int(len(all_good) * (1 - VAL_SPLIT))
    train_good = all_good[:split_idx]
    val_good = all_good[split_idx:]

    # BAD images — only test bad exist
    random.shuffle(bad_images)
    split_idx = int(len(bad_images) * (1 - VAL_SPLIT))
    train_bad = bad_images[:split_idx]
    val_bad = bad_images[split_idx:]

    print(f"[{obj_name}] good_train={len(train_good)}, good_val={len(val_good)}, "
          f"bad_train={len(train_bad)}, bad_val={len(val_bad)}")

    # Copy GOOD training images
    for img in train_good:
        out_img = OUT / "images" / "train" / f"{obj_name}_good_{img.name}"
        out_lbl = OUT / "labels" / "train" / f"{obj_name}_good_{img.stem}.txt"
        shutil.copy(img, out_img)
        write_yolo_label(out_lbl, 1, 1, cls_good)

    # Copy GOOD validation images
    for img in val_good:
        out_img = OUT / "images" / "val" / f"{obj_name}_good_{img.name}"
        out_lbl = OUT / "labels" / "val" / f"{obj_name}_good_{img.stem}.txt"
        shutil.copy(img, out_img)
        write_yolo_label(out_lbl, 1, 1, cls_good)

    # Copy BAD training images
    for img in train_bad:
        out_img = OUT / "images" / "train" / f"{obj_name}_bad_{img.name}"
        out_lbl = OUT / "labels" / "train" / f"{obj_name}_bad_{img.stem}.txt"
        shutil.copy(img, out_img)
        write_yolo_label(out_lbl, 1, 1, cls_bad)

    # Copy BAD validation images
    for img in val_bad:
        out_img = OUT / "images" / "val" / f"{obj_name}_bad_{img.name}"
        out_lbl = OUT / "labels" / "val" / f"{obj_name}_bad_{img.stem}.txt"
        shutil.copy(img, out_img)
        write_yolo_label(out_lbl, 1, 1, cls_bad)


def main():
    for i, cls_name in enumerate(CLASSES):
        process_class(cls_name, i)

    print("\n✔ DONE — YOLO dataset ready at:", OUT)


if __name__ == "__main__":
    main()
