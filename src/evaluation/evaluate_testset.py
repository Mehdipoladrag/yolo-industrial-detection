import glob
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os

MODEL_PATH = "runs/classify/train/weights/best.pt"
TEST_DIR = "data/processed/test"

model = YOLO(MODEL_PATH)

y_true = []
y_pred = []

classes = ["good", "bad"]

print("Running evaluation on test set...\n")

for cls in classes:
    folder = os.path.join(TEST_DIR, cls)
    images = glob.glob(os.path.join(folder, "*"))

    for img in images:
        result = model(img)[0]

        pred_idx = result.probs.top1
        pred_label = result.names[pred_idx]

        y_true.append(cls)
        y_pred.append(pred_label)

# Convert to numpy
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Print report
print("Classification Report:")
print(classification_report(y_true, y_pred, labels=classes))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=classes)

print("\nConfusion Matrix:")
print(cm)
