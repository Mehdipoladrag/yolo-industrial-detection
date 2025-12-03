import torch
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel

add_safe_globals([DetectionModel])

from ultralytics import YOLO
from .configs import MODEL_PATH


class ModelLoader:
    def __init__(self):
        self.model = None

    def load(self):
        if self.model is None:
            print("Loading model:", MODEL_PATH)
            self.model = YOLO(MODEL_PATH)
        return self.model

    def predict(self, image_path):
        model = self.load()
        return model.predict(image_path, save=False)[0]
