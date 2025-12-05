import os
import cv2
import time
from src.annotator.object_annotator import ObjectAnnotator

class Main:
    @staticmethod
    def process_folder(folder_path, output_folder, annotator, limit=None):
        os.makedirs(output_folder, exist_ok=True)

        images = os.listdir(folder_path)

        if limit is not None:
            images = images[:limit]

        print(f"Found {len(images)} items in folder...")

        count = 0
        start_time = time.time()  

        for filename in images:
            input_path = os.path.join(folder_path, filename)

            img_test = cv2.imread(input_path)
            if img_test is None:
                print(f"Skipping non-image file: {filename}")
                continue

            output_path = os.path.join(output_folder, filename)

            print(f"[INFO] Processing {filename}")
            annotator.annotate(input_path, output_path)
            count += 1

        end_time = time.time()   

        total_time = end_time - start_time
        print(f"\n[DONE] — Processed {count} valid images.")
        print(f"TOTAL TIME: {total_time:.2f} seconds")

    @staticmethod
    def run():
        annotator = ObjectAnnotator(
            min_conf=0.05,
            good_threshold=0.20,
            cover_ratio=0.95,
        )

        folder_path = "data/mvtec/bottle/test/mixed"
        output_folder = "output"

        Main.process_folder(folder_path, output_folder, annotator, limit=100)


if __name__ == "__main__":
    Main.run()
