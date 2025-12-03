import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(
    ROOT_DIR, 
    "runs", "detect", "train5", "weights", "best.pt"
)
IMG_SIZE = 640
DEFAULT_THRESHOLD = 0.75
