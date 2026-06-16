"""Shared helpers for the OpenCV + MediaPipe face demo.

Provides a single input abstraction so detection, landmark, capture and
recognition scripts all consume frames the same way, plus a Haar-cascade
based face detector.
"""

from __future__ import annotations

import os
from typing import Iterator, Optional

import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
QUIT_KEYS = {ord("q"), 27}  # 'q' or ESC

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(HERE, "dataset")
MODEL_DIR = os.path.join(HERE, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "lbph.yml")
MODEL_LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")
DATASET_LABELS_PATH = os.path.join(DATASET_DIR, "labels.json")


def is_image_path(source: str) -> bool:
    return os.path.splitext(source)[1].lower() in IMAGE_EXTENSIONS


def _resolve_source(source: str):
    """Map a CLI source string to something cv2.VideoCapture understands.

    ``"0"``/``"1"`` -> webcam index; any other string -> file path.
    """
    if source.isdigit():
        return int(source)
    return source


def iter_frames(source: str) -> Iterator["cv2.Mat"]:
    """Yield BGR frames from an image file, a video file, or a webcam.

    - Image: yields the single frame once.
    - Video/webcam: yields frames until the stream ends or the user quits.
    """
    if is_image_path(source):
        frame = cv2.imread(source)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {source}")
        yield frame
        return

    cap = cv2.VideoCapture(_resolve_source(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def load_face_cascade() -> "cv2.CascadeClassifier":
    """Load the frontal-face Haar cascade bundled with OpenCV."""
    path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade from {path}")
    return cascade


def detect_faces(cascade, frame, scale_factor: float = 1.1, min_neighbors: int = 5):
    """Return a list of (x, y, w, h) face boxes for a BGR frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(60, 60),
    )
    return list(faces)


def show_or_quit(window: str, frame, is_stream: bool, wait_ms: Optional[int] = None) -> bool:
    """Display a frame. Return True if the user requested to quit.

    For streams we poll briefly; for a single image we block until a key.
    """
    cv2.imshow(window, frame)
    delay = wait_ms if wait_ms is not None else (1 if is_stream else 0)
    key = cv2.waitKey(delay) & 0xFF
    return key in QUIT_KEYS
