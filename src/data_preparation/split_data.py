import shutil
import random
from pathlib import Path


class DatasetSplitter:
    """Split converted MVTec dataset into train/val/test WITHOUT deleting original images."""

    def __init__(self, processed_dir="data/processed", seed=42):
        self.base = Path(processed_dir)
        self.seed = seed
        random.seed(seed)

        # INPUT folders (from convert)
        self.input_good = self.base / "converted_good"
        self.input_bad = self.base / "converted_bad"

        # OUTPUT folders for YOLO format
        self.output = {
            "train_good": self.base / "train" / "good",
            "val_good": self.base / "val" / "good",
            "test_good": self.base / "test" / "good",

            "train_bad": self.base / "train" / "bad",
            "val_bad": self.base / "val" / "bad",
            "test_bad": self.base / "test" / "bad",
        }

        # Create input merge folders once
        self.input_good.mkdir(parents=True, exist_ok=True)
        self.input_bad.mkdir(parents=True, exist_ok=True)

    def merge_original(self):
        """Merge all good/bad images into input_good / input_bad BEFORE split."""
        print("🔄 Merging all GOOD images...")
        g1 = list((self.base / "train" / "good").glob("*"))
        g2 = list((self.base / "test" / "good").glob("*"))
        for f in g1 + g2:
            shutil.copy(f, self.input_good / f.name)

        print("🔄 Merging all BAD images...")
        b1 = list((self.base / "test" / "bad").glob("*"))
        for f in b1:
            shutil.copy(f, self.input_bad / f.name)

    @staticmethod
    def _split_list(items, ratio=(0.8, 0.1, 0.1)):
        n = len(items)
        n_train = int(n * ratio[0])
        n_val = int(n * ratio[1])
        random.shuffle(items)
        return (
            items[:n_train],
            items[n_train:n_train + n_val],
            items[n_train + n_val:]
        )

    @staticmethod
    def _copy_files(files, dst_folder):
        dst_folder.mkdir(parents=True, exist_ok=True)
        for src in files:
            shutil.copy(src, dst_folder / src.name)

    def split(self):
        # Merge convert output once
        self.merge_original()

        print("[INFO]Collecting GOOD images...")
        good_files = list(self.input_good.glob("*"))

        print("[INFO]Collecting BAD images...")
        bad_files = list(self.input_bad.glob("*"))

        print(f"[INFO]Found {len(good_files)} GOOD, {len(bad_files)} BAD")

        train_g, val_g, test_g = self._split_list(good_files)
        train_b, val_b, test_b = self._split_list(bad_files)

        print("Copying to YOLO folders...")

        # GOOD
        self._copy_files(train_g, self.output["train_good"])
        self._copy_files(val_g, self.output["val_good"])
        self._copy_files(test_g, self.output["test_good"])

        # BAD
        self._copy_files(train_b, self.output["train_bad"])
        self._copy_files(val_b, self.output["val_bad"])
        self._copy_files(test_b, self.output["test_bad"])

        print("\nSplit Done!")
        print(f"Train: {len(train_g)} good | {len(train_b)} bad")
        print(f"Val:   {len(val_g)} good | {len(val_b)} bad")
        print(f"Test:  {len(test_g)} good | {len(test_b)} bad")


if __name__ == "__main__":
    splitter = DatasetSplitter()
    splitter.split()
