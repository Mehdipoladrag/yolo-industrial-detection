import os
import shutil
from pathlib import Path
import random
import cv2

RAW = Path("data/raw/mvtec_ad")
OUT = Path("data/detection_objects")

CLASSES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper"
]


class DatasetPreparer:
    def prepare_folders(self):
        for split in ["train", "val"]:
            (OUT/"images"/split).mkdir(parents=True, exist_ok=True)
            (OUT/"labels"/split).mkdir(parents=True, exist_ok=True)

class CreateDataset:
    def create_dataset(self):
        preparer = DatasetPreparer()
        preparer.prepare_folders()
        print("Creating dataset for Object Detection...\n")
        class_map = {cls: i for i, cls in enumerate(CLASSES)}
        print("Class Map:", class_map)

        for cls in CLASSES:
            print(f"Processing: {cls}")

            all_imgs = list((RAW / cls / "train" / "good").glob("*.png")) + \
                    [img for defect in (RAW/cls/"test").glob("*") if defect.name!="good"
                            for img in defect.glob("*.png")] + \
                        list((RAW/cls/"test"/"good").glob("*.png"))

            random.shuffle(all_imgs)

            split_idx = int(len(all_imgs) * 0.8)
            train_imgs = all_imgs[:split_idx]
            val_imgs = all_imgs[split_idx:]

            for split, images in [("train", train_imgs), ("val", val_imgs)]:
                for img_path in images:
                    out_img = OUT/"images"/split/img_path.name
                    shutil.copy(img_path, out_img)

                    # YOLO bounding box 100% image
                    with open(OUT/"labels"/split/f"{img_path.stem}.txt", "w") as f:
                        cls_id = class_map[cls]
                        f.write(f"{cls_id} 0.5 0.5 1 1\n")

        print("\nDataset created successfully!")


if __name__ == "__main__":
    dataset_creator = CreateDataset()
    dataset_creator.create_dataset()