import torch
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel

add_safe_globals([DetectionModel])

from ultralytics import YOLO
from .configs import MODEL_PATH, IMG_SIZE

class ModelLoader:
    def __init__(self):
        self.model = None

    def load(self):
        if self.model is None:
            print("Loading model:", MODEL_PATH)
            self.model = YOLO(MODEL_PATH)
        return self.model

    def predict(self, image_path, conf: float = 0.001):

        model = self.load()
        results = model.predict(
            source=image_path,
            save=False,
            conf=conf,     
            imgsz=640,
            verbose=True,
        )
        return results[0]
