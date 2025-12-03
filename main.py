from src.annotator.object_annotator import ObjectAnnotator

def main():
    annotator = ObjectAnnotator(good_threshold=0.75)

    annotator.annotate(
        image_path="data/detection_objects/images/val/0004.png",
        output_path="output/final.jpg"
    )

if __name__ == "__main__":
    main()