import os
import glob
from ultralytics import YOLO
from src.utils.save_to_txt import save_predictions

MODEL_PATH = "runs/classify/train/weights/best.pt"
SOURCE_FOLDER = "data/processed/train"

YOLO_OUTPUT = "runs/classify/custom_predict"
OUTPUT_FOLDER = "src/output"
TXT_NAME = "predictions.txt"

model = YOLO(MODEL_PATH)

good_images = glob.glob(os.path.join(SOURCE_FOLDER, "good", "*"))
bad_images = glob.glob(os.path.join(SOURCE_FOLDER, "bad", "*"))

all_images = [(img, "good") for img in good_images] + \
             [(img, "bad") for img in bad_images]

predictions = []

for img_path, true_lbl in all_images:
    result = model(img_path, save=True, project=YOLO_OUTPUT, exist_ok=True)[0]

    idx = result.probs.top1
    pred_label = result.names[idx]
    conf = float(result.probs.top1conf)

    print(f"{img_path} → {pred_label} ({conf:.4f})")

    predictions.append({
        "image": img_path,
        "label": pred_label,
        "confidence": conf,
        "true_label": true_lbl
    })

save_predictions(OUTPUT_FOLDER, TXT_NAME, predictions)
