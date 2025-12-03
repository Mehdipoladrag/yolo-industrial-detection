import os
import cv2
from models.loader import ModelLoader


class ObjectAnnotator:
    def __init__(self, good_threshold: float = 0.75, min_conf: float = 0.60):

        self.model_loader = ModelLoader()
        self.good_threshold = good_threshold
        self.min_conf = min_conf

    def annotate(self, image_path: str, output_path: str) -> None:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        # اجرای مدل
        results = self.model_loader.predict(image_path)

        boxes = results.boxes

        # YOLO11 stores class names in results[0].names  ← مهم‌ترین نکته!
        names = results[0].names

        for box in boxes:
            conf = float(box.conf[0])
            if conf < self.min_conf:
                continue
            self._draw_label(img, box, names)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
        print(f"✔ Saved: {output_path}")

    def _draw_label(self, img, box, names):

        h, w = img.shape[:2]

        # coords
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # class + score
        cls_id = int(box.cls[0])

        # handle dict or list
        class_name = names[cls_id] if not isinstance(names, dict) else names.get(cls_id, "Object")

        conf = float(box.conf[0])
        is_good = conf >= self.good_threshold
        status = "GOOD" if is_good else "BAD"

        label = f"{class_name} ({status})"

        # color
        color = (0, 255, 0) if is_good else (0, 0, 255)

        # draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # text style
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)

        # --- place label *INSIDE* the box ---
        text_x = x1 + 5
        text_y = y1 + th + 5

        # background inside box
        cv2.rectangle(img,
            (x1, y1),
            (x1 + tw + 10, y1 + th + 10),
            color, -1
        )

        # text
        cv2.putText(
            img,
            label,
            (text_x, text_y),
            font, scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )
