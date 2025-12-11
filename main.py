import cv2
from src.annotator.object_annotator import ObjectAnnotator

def main():
    url = "http://10.175.168.130:8080/video"
    cap = cv2.VideoCapture(url)

    annotator = ObjectAnnotator(
        min_conf=0.05,
        good_threshold=0.20,
        cover_ratio=0.95,
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR]NO FRAME")
            continue

        annotated, status = annotator.annotate_frame(frame)

        if status == "GOOD":
            color = (0, 255, 0)
        elif status == "BAD":
            color = (0, 0, 255)
        else:
            color = (0, 255, 255)  

        cv2.putText(
            annotated,
            status,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3,
        )

        cv2.imshow("Realtime QA", annotated)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
