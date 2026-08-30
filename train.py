"""
train.py

Fine-tunes a YOLOv8 classification model (yolov8n-cls) on the prepared
waste dataset (see prepare_dataset.py for the expected folder layout).

Run this after prepare_dataset.py has produced:
    dataset/train/<class>/*.jpg
    dataset/val/<class>/*.jpg
    dataset/test/<class>/*.jpg

Requires: pip install ultralytics
"""

from ultralytics import YOLO

DATASET_DIR = "dataset"       # root folder containing train/ val/ test/
EPOCHS = 25
IMG_SIZE = 224                # standard for yolov8-cls
BATCH_SIZE = 32
MODEL_SIZE = "yolov8n-cls.pt" # nano = fastest to train, good for a resume project on limited time/compute


def main():
    print(f"Loading base model: {MODEL_SIZE}")
    model = YOLO(MODEL_SIZE)

    print("Starting training...")
    results = model.train(
        data=DATASET_DIR,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project="runs",
        name="waste_cls",
        patience=5,       # early stopping if val accuracy plateaus
    )

    print("\nTraining complete.")
    print(f"Best weights saved to: runs/waste_cls/weights/best.pt")

    # Quick validation on the held-out test set
    print("\nEvaluating on validation set...")
    metrics = model.val()
    print(metrics)


if __name__ == "__main__":
    main()
