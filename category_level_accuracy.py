"""
category_level_accuracy.py

Uses the confusion matrix from validation + the 30-class -> disposal-category
mapping to compute a second, more forgiving accuracy metric:

    "Category-level accuracy" = of all predictions, what fraction landed in
    the CORRECT disposal category (recyclable / organic / non-recyclable /
    hazardous), even if the specific item class (e.g. aluminum_food_cans vs
    steel_food_cans) was wrong?

This matters because for the actual use case (telling someone which bin to
use), a mistake like "aluminum_food_cans" -> "steel_food_cans" is harmless:
both are recyclable, so the user still gets the right disposal instruction.
Plain top-1 accuracy (83.7%) doesn't capture that; this metric does.

Run this AFTER training (needs runs/waste_cls/weights/best.pt to exist).
"""

from ultralytics import YOLO
from category_mapping import get_category

MODEL_PATH = "runs/waste_cls/weights/best.pt"
DATASET_DIR = "dataset"


def main():
    model = YOLO(MODEL_PATH)

    print("Running validation to get the confusion matrix...")
    metrics = model.val(data=DATASET_DIR)

    cm = metrics.confusion_matrix.matrix  # rows = predicted, cols = true (Ultralytics convention)
    class_names = list(model.names.values())
    n = len(class_names)

    total = 0
    correct_class = 0
    correct_category = 0

    for true_idx in range(n):
        true_class = class_names[true_idx]
        true_category = get_category(true_class)

        for pred_idx in range(n):
            pred_class = class_names[pred_idx]
            count = cm[pred_idx][true_idx]
            if count == 0:
                continue

            total += count
            if pred_idx == true_idx:
                correct_class += count

            pred_category = get_category(pred_class)
            if pred_category == true_category:
                correct_category += count

    class_acc = correct_class / total
    category_acc = correct_category / total

    print(f"\nTotal predictions counted: {int(total)}")
    print(f"Top-1 class accuracy:      {class_acc:.1%}   (specific item, e.g. 'aluminum_food_cans')")
    print(f"Category-level accuracy:   {category_acc:.1%}   (recyclable/organic/non-recyclable/hazardous)")
    print(f"\nGain from category-level grouping: +{(category_acc - class_acc):.1%}")


if __name__ == "__main__":
    main()
