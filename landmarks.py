"""Facial landmark extraction and mesh visualization using MediaPipe Face Mesh.

Examples:
    python landmarks.py --source 0
    python landmarks.py --source face.jpg --save mesh.jpg
"""

from __future__ import annotations

import argparse

import cv2
import mediapipe as mp

from common import is_image_path, iter_frames, show_or_quit

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def extract_and_draw(face_mesh, frame):
    """Run Face Mesh on a BGR frame and draw tessellation + contours in place.

    Returns the number of faces for which landmarks were found.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return 0

    for landmarks in result.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
        )
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style(),
        )
    return len(result.multi_face_landmarks)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize MediaPipe face landmarks.")
    parser.add_argument("--source", default="0", help="webcam index, image path, or video path")
    parser.add_argument("--max-faces", type=int, default=2)
    parser.add_argument("--save", help="optional output path (image source only)")
    return parser.parse_args()


def main():
    args = parse_args()
    is_stream = not is_image_path(args.source)
    window = "Face landmarks"

    with mp_face_mesh.FaceMesh(
        static_image_mode=not is_stream,
        max_num_faces=args.max_faces,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        for frame in iter_frames(args.source):
            extract_and_draw(face_mesh, frame)

            if args.save and not is_stream:
                cv2.imwrite(args.save, frame)
                print(f"Saved landmark image to {args.save}")

            if show_or_quit(window, frame, is_stream):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
