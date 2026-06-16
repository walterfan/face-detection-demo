"""Capture labeled grayscale face samples into the dataset directory.

Samples are stored as dataset/<label>_<username>/NNN.png and the
label -> username mapping is kept in dataset/labels.json.

Examples:
    python capture.py --username walter --label 1
    python capture.py --username alice --label 2 --count 30 --source 0
"""

from __future__ import annotations

import argparse
import json
import os

import cv2

from common import (
    DATASET_DIR,
    DATASET_LABELS_PATH,
    detect_faces,
    iter_frames,
    load_face_cascade,
)

FACE_SIZE = (200, 200)


def load_labels() -> dict:
    if os.path.exists(DATASET_LABELS_PATH):
        with open(DATASET_LABELS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_labels(labels: dict) -> None:
    os.makedirs(DATASET_DIR, exist_ok=True)
    with open(DATASET_LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)


def next_index(user_dir: str) -> int:
    if not os.path.isdir(user_dir):
        return 0
    existing = [n for n in os.listdir(user_dir) if n.lower().endswith(".png")]
    return len(existing)


def parse_args():
    parser = argparse.ArgumentParser(description="Capture grayscale face samples for a user.")
    parser.add_argument("--username", required=True, help="human-readable name for this identity")
    parser.add_argument("--label", required=True, type=int, help="numeric label id for this identity")
    parser.add_argument("--count", type=int, default=30, help="number of samples to collect")
    parser.add_argument("--source", default="0", help="webcam index, image path, or video path")
    return parser.parse_args()


def main():
    args = parse_args()
    cascade = load_face_cascade()

    user_dir = os.path.join(DATASET_DIR, f"{args.label}_{args.username}")
    os.makedirs(user_dir, exist_ok=True)

    labels = load_labels()
    labels[str(args.label)] = args.username
    save_labels(labels)

    start = next_index(user_dir)
    saved = start
    target = start + args.count
    window = "Capture (q to stop)"
    print(f"Capturing up to {args.count} samples for '{args.username}' (label {args.label})...")

    for frame in iter_frames(args.source):
        faces = detect_faces(cascade, frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use the largest face only, to avoid mixing identities in one sample.
        if faces:
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            face = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)
            path = os.path.join(user_dir, f"{saved:03d}.png")
            cv2.imwrite(path, face)
            saved += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(
            frame, f"{saved}/{target}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
        )
        cv2.imshow(window, frame)
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27) or saved >= target:
            break

    cv2.destroyAllWindows()
    print(f"Saved {saved - start} new samples to {user_dir}")
    print(f"Dataset now at {saved} total samples for this user.")


if __name__ == "__main__":
    main()
