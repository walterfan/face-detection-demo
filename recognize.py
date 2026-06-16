"""Recognize faces with a trained LBPH model (1:1 verification).

Loads model/lbph.yml + model/labels.json, detects faces from the source,
and annotates each with the matched username and confidence. Note that
LBPH confidence is a distance: LOWER means a better match.

Examples:
    python recognize.py --source 0
    python recognize.py --source group.jpg --threshold 70
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2

from common import (
    MODEL_LABELS_PATH,
    MODEL_PATH,
    detect_faces,
    is_image_path,
    iter_frames,
    load_face_cascade,
    show_or_quit,
)

FACE_SIZE = (200, 200)


def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(MODEL_LABELS_PATH):
        print(
            f"Error: model not found. Expected {MODEL_PATH} and {MODEL_LABELS_PATH}.\n"
            "Run train.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    with open(MODEL_LABELS_PATH, "r", encoding="utf-8") as f:
        label_to_name = json.load(f)
    return recognizer, label_to_name


def parse_args():
    parser = argparse.ArgumentParser(description="Recognize faces using a trained LBPH model.")
    parser.add_argument("--source", default="0", help="webcam index, image path, or video path")
    parser.add_argument("--threshold", type=float, default=70.0,
                        help="max LBPH distance to accept a match (lower is stricter)")
    return parser.parse_args()


def main():
    args = parse_args()
    recognizer, label_to_name = load_model()
    cascade = load_face_cascade()
    is_stream = not is_image_path(args.source)
    window = "Face recognition"

    for frame in iter_frames(args.source):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for (x, y, w, h) in detect_faces(cascade, frame):
            face = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)
            label, confidence = recognizer.predict(face)

            if confidence <= args.threshold:
                username = label_to_name.get(str(label), f"label {label}")
                color = (0, 255, 0)
                caption = f"{username} ({confidence:.1f})"
                print(f"Match: label={label} user={username} confidence={confidence:.1f}")
            else:
                color = (0, 0, 255)
                caption = f"unknown ({confidence:.1f})"
                print(f"Unknown: best label={label} confidence={confidence:.1f} (> {args.threshold})")

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, caption, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if show_or_quit(window, frame, is_stream):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
