"""Quick, reproducible LBPH sanity check on the public Olivetti/ORL dataset.

No webcam or manual download required: scikit-learn fetches the Olivetti
faces (40 people x 10 grayscale 64x64 images). We split per subject into
train/test, train an LBPH recognizer, and report recognition accuracy.

Example:
    python verify_olivetti.py --train-per-person 8
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

try:
    from sklearn.datasets import fetch_olivetti_faces
except ImportError:  # pragma: no cover - dependency hint
    raise SystemExit(
        "scikit-learn is required for this script. Install it with:\n"
        "    pip install scikit-learn"
    )


def to_uint8(image_float):
    return (image_float * 255).astype(np.uint8)


def split(data, train_per_person: int):
    """Per-subject split: first N images train, the rest test."""
    train_x, train_y, test_x, test_y = [], [], [], []
    for label in np.unique(data.target):
        idx = np.where(data.target == label)[0]
        for rank, i in enumerate(idx):
            img = to_uint8(data.images[i])
            if rank < train_per_person:
                train_x.append(img)
                train_y.append(int(label))
            else:
                test_x.append(img)
                test_y.append(int(label))
    return train_x, train_y, test_x, test_y


def parse_args():
    parser = argparse.ArgumentParser(description="LBPH accuracy check on Olivetti faces.")
    parser.add_argument("--train-per-person", type=int, default=8,
                        help="images per subject used for training (1-9)")
    return parser.parse_args()


def main():
    args = parse_args()
    data = fetch_olivetti_faces(shuffle=False)
    train_x, train_y, test_x, test_y = split(data, args.train_per_person)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(train_x, np.array(train_y))

    correct = 0
    for img, truth in zip(test_x, test_y):
        pred, _confidence = recognizer.predict(img)
        if pred == truth:
            correct += 1

    total = len(test_y)
    accuracy = correct / total if total else 0.0
    print(f"Train samples: {len(train_y)} | Test samples: {total}")
    print(f"Accuracy: {correct}/{total} = {accuracy:.1%}")


if __name__ == "__main__":
    main()
