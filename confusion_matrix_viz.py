"""
confusion_matrix_viz.py

Loads the confusion matrix generated during model.val() and visualizes it,
so you can see exactly which classes the model confuses with each other.

Run this AFTER training (needs runs/waste_cls/weights/best.pt to exist).
"""

from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

MODEL_PATH = "runs/waste_cls/weights/best.pt"
DATASET_DIR = "dataset"


def main():
    model = YOLO(MODEL_PATH)

    print("Running validation to generate confusion matrix...")
    metrics = model.val(data=DATASET_DIR)

    cm = metrics.confusion_matrix.matrix  # shape: (num_classes+1, num_classes+1) -- includes background
    class_names = list(model.names.values())

    # Ultralytics saves its own plotted confusion matrix automatically to
    # runs/classify/val/confusion_matrix.png and confusion_matrix_normalized.png
    # This script prints the top confusions in plain text as well, which is
    # often more useful than staring at a 30x30 grid.

    print("\nTop confused class pairs (excluding the diagonal / correct predictions):")
    n = len(class_names)
    confusions = []
    for i in range(n):
        for j in range(n):
            if i != j and cm[i][j] > 0:
                confusions.append((cm[i][j], class_names[j], class_names[i]))
                # cm[i][j] = number of times true class j was predicted as class i (Ultralytics convention: rows=predicted, cols=true)

    confusions.sort(reverse=True)
    for count, true_class, predicted_as in confusions[:15]:
        print(f"  {true_class:30s} -> predicted as {predicted_as:30s}  ({int(count)} times)")

    print(f"\nFull confusion matrix images saved by Ultralytics to:")
    print(f"  runs/classify/val/confusion_matrix.png")
    print(f"  runs/classify/val/confusion_matrix_normalized.png")


if __name__ == "__main__":
    main()
