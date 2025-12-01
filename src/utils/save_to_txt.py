import os

def save_predictions(output_folder, file_name, predictions):


    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, file_name)

    correct = 0
    wrong = 0

    for p in predictions:
        if "true_label" in p:   
            if p["label"] == p["true_label"]:
                correct += 1
            else:
                wrong += 1

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("────────── YOLO Prediction Results ──────────\n\n")

        for p in predictions:
            f.write(f"Image: {p['image']}\n")
            f.write(f"Predicted: {p['label']}\n")
            f.write(f"Confidence: {p['confidence']:.4f}\n")
            if "true_label" in p:
                f.write(f"True Label: {p['true_label']}\n")
            f.write("---------------------------------------------\n")

        total = len(predictions)
        accuracy = (correct / total * 100) if total > 0 else 0

        f.write("\n================ SUMMARY ================\n")
        f.write(f"Total Images: {total}\n")
        if correct + wrong > 0:
            f.write(f"Correct Predictions: {correct}\n")
            f.write(f"Wrong Predictions:   {wrong}\n")
            f.write(f"Accuracy:            {accuracy:.2f}%\n")
        f.write("=========================================\n")

    print(f"\n[DONE] File Created: {out_path}")
