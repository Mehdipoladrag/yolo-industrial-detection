import cv2 as cv
import os
from models.loader import ModelLoader


class ObjectAnnotator:

    def __init__(
        self,
        min_conf: float = 0.01,
        good_threshold: float = 0.07,
        cover_ratio: float = 1.0,
        box_thickness: int = 3,
        font_scale: float = 0.7,
        font_thickness: int = 2,
    ):
        self.model_loader = ModelLoader()
        self.min_conf = min_conf
        self.good_threshold = good_threshold
        self.cover_ratio = cover_ratio
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.font_thickness = font_thickness


    def annotate(self, image_path: str, output_path: str):

        img = cv.imread(image_path)
        if img is None:
            raise FileNotFoundError(image_path)

        H, W = img.shape[:2]

        results = self.model_loader.predict(image_path)
        names = results.names
        boxes = results.boxes

        if boxes is None or len(boxes) == 0:
            print("[ERROR] No detections → saving original image")
            self._save(output_path, img)
            return

        best = None
        best_conf = -1.0

        for box in boxes:
            conf = float(box.conf.item())
            if conf > best_conf:
                best = box
                best_conf = conf

        if best_conf < self.min_conf:
            print("[ERROR] Low confidence → saving original image")
            self._save(output_path, img)
            return

        x1, y1, x2, y2 = map(int, best.xyxy[0])
        cls_id = int(best.cls.item())
        raw_name = names[cls_id]

        label, status = self._make_label(raw_name, best_conf)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        bw = int(W * self.cover_ratio)
        bh = int(H * self.cover_ratio)

        big_x1 = max(0, cx - bw)
        big_y1 = max(0, cy - bh)
        big_x2 = min(W - 1, cx + bw)
        big_y2 = min(H - 1, cy + bh)

        # Draw annotation box
        self._draw_box(img, big_x1, big_y1, big_x2, big_y2, label, status)

        self._save(output_path, img)
        print(f"[Done] Saved: {output_path}")


    def _make_label(self, raw_name, conf):
        """Generate label text and determine GOOD/BAD status."""
        name = raw_name.replace("_", " ")

        if raw_name.endswith("_good"):
            return f"{name} (GOOD) {conf:.2f}", "GOOD"

        if raw_name.endswith("_bad"):
            return f"{name} (BAD) {conf:.2f}", "BAD"

        # If class does not explicitly encode good/bad
        status = "GOOD" if conf >= self.good_threshold else "BAD"
        return f"{name} ({status}) {conf:.2f}", status


    def _draw_box(self, img, x1, y1, x2, y2, text, status):
        """Draw bounding box + label."""
        color = (0, 255, 0) if status == "GOOD" else (0, 0, 255)
        
        cv.rectangle(img, (x1, y1), (x2, y2), color, self.box_thickness)

        font = cv.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv.getTextSize(text, font, self.font_scale, self.font_thickness)

        bg_x2 = x1 + tw + 12
        bg_y2 = y1 + th + 12

        cv.rectangle(img, (x1, y1), (bg_x2, bg_y2), color, -1)

        cv.putText(
            img,
            text,
            (x1 + 6, y1 + th + 4),
            font,
            self.font_scale,
            (255, 255, 255),
            self.font_thickness,
            cv.LINE_AA,
        )
    def annotate_frame(self, frame):
        """Realtime annotate without saving to file."""
        H, W = frame.shape[:2]

        import tempfile, cv2, os
        tmp_in = tempfile.mktemp(suffix=".jpg")
        cv2.imwrite(tmp_in, frame)

        results = self.model_loader.predict(tmp_in)
        names = results.names
        boxes = results.boxes

        if boxes is None or len(boxes) == 0:
            return frame, "NO DETECTION"

        best = None
        best_conf = -1
        for box in boxes:
            conf = float(box.conf.item())
            if conf > best_conf:
                best = box
                best_conf = conf

        if best_conf < self.min_conf:
            return frame, "LOW_CONF"

        x1, y1, x2, y2 = map(int, best.xyxy[0])
        cls_id = int(best.cls.item())
        raw_name = names[cls_id]

        label, status = self._make_label(raw_name, best_conf)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        bw = int(W * self.cover_ratio)
        bh = int(H * self.cover_ratio)

        bx1 = max(0, cx - bw)
        by1 = max(0, cy - bh)
        bx2 = min(W - 1, cx + bw)
        by2 = min(H - 1, cy + bh)

        self._draw_box(frame, bx1, by1, bx2, by2, label, status)

        return frame, status


    def _save(self, path, img):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv.imwrite(path, img)
