"""Face detection from an image, video file, or webcam using a Haar cascade.

Examples:
    python detect.py --source 0                 # webcam
    python detect.py --source face.jpg          # image, press any key to close
    python detect.py --source clip.mp4 --shape circle
    python detect.py --source face.jpg --save out.jpg
"""

from __future__ import annotations

import argparse

import cv2

from common import detect_faces, is_image_path, iter_frames, load_face_cascade, show_or_quit


def annotate(frame, faces, shape: str):
    for (x, y, w, h) in faces:
        if shape == "circle":
            center = (x + w // 2, y + h // 2)
            radius = max(w, h) // 2
            cv2.circle(frame, center, radius, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame


def parse_args():
    parser = argparse.ArgumentParser(description="Detect faces from image/video/webcam.")
    parser.add_argument("--source", default="0", help="webcam index (e.g. 0), image path, or video path")
    parser.add_argument("--shape", choices=["rectangle", "circle"], default="rectangle")
    parser.add_argument("--save", help="optional output path (image source only)")
    return parser.parse_args()


def main():
    args = parse_args()
    cascade = load_face_cascade()
    is_stream = not is_image_path(args.source)
    window = "Face detection"

    for frame in iter_frames(args.source):
        faces = detect_faces(cascade, frame)
        annotate(frame, faces, args.shape)

        if args.save and not is_stream:
            cv2.imwrite(args.save, frame)
            print(f"Saved annotated image to {args.save}")

        if show_or_quit(window, frame, is_stream):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
