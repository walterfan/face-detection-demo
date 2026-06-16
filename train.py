"""Train an LBPH face recognizer from the captured dataset.

Reads grayscale samples from dataset/<label>_<username>/*.png, trains
cv2.face.LBPHFaceRecognizer_create(), and writes:
  - model/lbph.yml      (the trained model)
  - model/labels.json   (label -> username mapping)

Example:
    python train.py
"""

from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

from common import (
    DATASET_DIR,
    DATASET_LABELS_PATH,
    MODEL_DIR,
    MODEL_LABELS_PATH,
    MODEL_PATH,
)


def load_dataset():
    """Return (images, labels, label_to_name) loaded from the dataset dir."""
    images, labels = [], []
    label_to_name: dict[str, str] = {}

    if os.path.exists(DATASET_LABELS_PATH):
        with open(DATASET_LABELS_PATH, "r", encoding="utf-8") as f:
            label_to_name = json.load(f)

    if not os.path.isdir(DATASET_DIR):
        return images, labels, label_to_name

    for entry in sorted(os.listdir(DATASET_DIR)):
        user_dir = os.path.join(DATASET_DIR, entry)
        if not os.path.isdir(user_dir) or "_" not in entry:
            continue
        label_str, username = entry.split("_", 1)
        if not label_str.isdigit():
            continue
        label = int(label_str)
        label_to_name.setdefault(str(label), username)

        for name in sorted(os.listdir(user_dir)):
            if not name.lower().endswith(".png"):
                continue
            img = cv2.imread(os.path.join(user_dir, name), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            images.append(img)
            labels.append(label)

    return images, labels, label_to_name


def main():
    images, labels, label_to_name = load_dataset()

    if not images:
        print(
            f"Error: no usable face samples found under {DATASET_DIR}.\n"
            "Run capture.py first to collect samples.",
            file=sys.stderr,
        )
        sys.exit(1)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels))

    os.makedirs(MODEL_DIR, exist_ok=True)
    recognizer.write(MODEL_PATH)
    with open(MODEL_LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(label_to_name, f, ensure_ascii=False, indent=2)

    print(f"Trained on {len(images)} samples across {len(set(labels))} identities.")
    print(f"Model saved to {MODEL_PATH}")
    print(f"Labels saved to {MODEL_LABELS_PATH}")


if __name__ == "__main__":
    main()
