from ultralytics import YOLO

def main():
    model = YOLO("models/yolo11s.pt")

    model.train(
        data="data/detection_objects/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,       
        workers=0,       
        pretrained=True
    )


if __name__ == "__main__":
    main()
