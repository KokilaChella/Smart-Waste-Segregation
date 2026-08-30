"""
prepare_dataset.py

Downloads the "Recyclable and Household Waste Classification Dataset"
(Alistair King, Kaggle) and reorganizes it into the folder structure
YOLOv8 classification mode expects:

    dataset/
        train/
            <class_name>/
                img1.jpg
                ...
        val/
            <class_name>/
                ...
        test/
            <class_name>/
                ...

Run this locally or in Google Colab (needs a kaggle.json API token).
Docs for getting your token: https://www.kaggle.com/docs/api
"""

import os
import random
import shutil
from pathlib import Path

# ---- CONFIG ----
KAGGLE_DATASET = "alistairking/recyclable-and-household-waste-classification"
RAW_DIR = Path("raw_dataset")          # where kaggle download lands
OUT_DIR = Path("dataset")              # YOLOv8-cls ready output
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15                      # must sum to 1.0
SEED = 42
# ----------------


def download_dataset():
    """Downloads and unzips the dataset via the Kaggle API/CLI."""
    RAW_DIR.mkdir(exist_ok=True)
    # Requires: pip install kaggle  +  ~/.kaggle/kaggle.json token in place
    os.system(f"kaggle datasets download -d {KAGGLE_DATASET} -p {RAW_DIR} --unzip")
    print(f"Downloaded and unzipped into: {RAW_DIR.resolve()}")


def find_image_root(raw_dir: Path) -> Path:
    """
    The Kaggle zip usually extracts to something like:
        raw_dataset/images/images/<class_name>/{default,real_world}/*.jpg
    This walks the tree to find the folder that actually contains the
    30 class subfolders, so the script doesn't break if the nesting
    differs slightly between downloads.
    """
    for path in raw_dir.rglob("*"):
        if path.is_dir():
            subdirs = [d for d in path.iterdir() if d.is_dir()]
            # Heuristic: a folder with ~30 subfolders is almost certainly
            # the class-folder root for this dataset.
            if 20 <= len(subdirs) <= 40:
                return path
    raise RuntimeError(
        "Could not auto-detect the class-folder root. "
        "Inspect raw_dataset/ manually and set the path directly."
    )


def collect_images_per_class(image_root: Path):
    """
    Returns { class_name: [list of image file paths] }.
    Merges the 'default' and 'real_world' subfolders into one pool per class,
    since for our purposes we want the model to generalize across both.
    """
    class_images = {}
    for class_dir in sorted(image_root.iterdir()):
        if not class_dir.is_dir():
            continue
        images = list(class_dir.rglob("*.jpg")) + list(class_dir.rglob("*.png")) + list(class_dir.rglob("*.jpeg"))
        if images:
            class_images[class_dir.name] = images
    return class_images


def split_and_copy(class_images: dict):
    random.seed(SEED)
    assert abs(TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT - 1.0) < 1e-6, "Splits must sum to 1.0"

    for split in ["train", "val", "test"]:
        (OUT_DIR / split).mkdir(parents=True, exist_ok=True)

    summary = {}
    for class_name, images in class_images.items():
        random.shuffle(images)
        n = len(images)
        n_train = int(n * TRAIN_SPLIT)
        n_val = int(n * VAL_SPLIT)

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split, split_images in splits.items():
            dest_dir = OUT_DIR / split / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img_path in split_images:
                shutil.copy2(img_path, dest_dir / img_path.name)

        summary[class_name] = {k: len(v) for k, v in splits.items()}

    return summary


def main():
    print("Step 1/3: Downloading dataset from Kaggle...")
    download_dataset()

    print("Step 2/3: Locating class folders...")
    image_root = find_image_root(RAW_DIR)
    print(f"  Found class-folder root: {image_root}")

    class_images = collect_images_per_class(image_root)
    print(f"  Found {len(class_images)} classes.")

    print("Step 3/3: Splitting into train/val/test and copying...")
    summary = split_and_copy(class_images)

    total = sum(sum(v.values()) for v in summary.values())
    print(f"\nDone. {total} images organized into: {OUT_DIR.resolve()}")
    print("\nPer-class counts (train/val/test):")
    for class_name, counts in summary.items():
        print(f"  {class_name:30s} {counts['train']:4d} / {counts['val']:4d} / {counts['test']:4d}")


if __name__ == "__main__":
    main()
