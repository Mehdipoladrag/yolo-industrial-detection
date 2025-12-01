import shutil
from pathlib import Path


class DatasetPaths:
    """Holds all dataset paths."""
    def __init__(self, raw_dir="data/raw/mvtec_ad", out_dir="data/processed"):
        self.raw_dir = Path(raw_dir)
        self.out_dir = Path(out_dir)


class DirectoryManager:
    """Creates YOLO directory structure."""
    SPLITS = ["train", "val", "test"]
    LABELS = ["good", "bad"]

    def __init__(self, base_output: Path):
        self.base_output = base_output

    def create(self):
        for split in self.SPLITS:
            for label in self.LABELS:
                (self.base_output / split / label).mkdir(parents=True, exist_ok=True)


class MVTecConverter:
    """Converts MVTec AD format into YOLO good/bad classification format."""
    
    CLASSES = [
        "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
        "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
        "transistor", "wood", "zipper"
    ]

    def __init__(self, paths: DatasetPaths):
        self.paths = paths
        self.raw = paths.raw_dir
        self.out = paths.out_dir
        self.dir_manager = DirectoryManager(self.out)

    def convert(self):
        print("Creating directory structure...")
        self.dir_manager.create()

        print("Converting MVTec dataset to YOLO good/bad format...\n")

        for cls_name in self.CLASSES:
            cls_path = self.raw / cls_name
            print(f"➡ Processing class: {cls_name}")

            # GOOD images from train
            train_good = cls_path / "train" / "good"
            for img in train_good.glob("*"):
                self._copy(img, self.out / "train" / "good", f"{cls_name}_train_{img.name}")

            # GOOD images from test
            test_good = cls_path / "test" / "good"
            for img in test_good.glob("*"):
                self._copy(img, self.out / "test" / "good", f"{cls_name}_test_{img.name}")

            # BAD images from all test defects
            for defect_folder in (cls_path / "test").iterdir():
                if defect_folder.name == "good":
                    continue
                for img in defect_folder.glob("*"):
                    self._copy(
                        img,
                        self.out / "test" / "bad",
                        f"{cls_name}_{defect_folder.name}_{img.name}"
                    )

        print("\nDONE! Dataset converted successfully.")

    @staticmethod
    def _copy(src, dst, new_name):
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst / new_name)


if __name__ == "__main__":
    paths = DatasetPaths()
    converter = MVTecConverter(paths)
    converter.convert()
