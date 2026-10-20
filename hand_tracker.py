#!/usr/bin/env python3
"""
Laptop Hand Tracking Proof of Concept

Uses the same Google MediaPipe Hand Landmarker model and 21-point landmark
layout intended for the HoloLens 2 project, but reads directly from the
laptop webcam.

Controls:
  Q / Esc  Quit
  S        Save a screenshot
  J        Toggle continuous JSON landmark output
  H        Toggle help text

The file latest_landmarks.json uses the same simple landmark structure that
can later be consumed by Unity/HoloLens code.
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
JSON_PATH = HERE / "latest_landmarks.json"
SCREENSHOT_DIR = HERE / "screenshots"

# MediaPipe hand landmark indices:
# 0 wrist
# 1-4 thumb
# 5-8 index
# 9-12 middle
# 13-16 ring
# 17-20 pinky
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (17, 0),
)

# BGR colors used by OpenCV.
LEFT_COLOR = (255, 220, 40)
RIGHT_COLOR = (50, 200, 255)
TEXT_COLOR = (245, 245, 245)
GOOD_COLOR = (80, 220, 80)
PANEL_COLOR = (20, 20, 20)


def ensure_model():
    """Download Google's official MediaPipe Hand Landmarker model if missing."""
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("MediaPipe hand model not found.")
    print("Downloading:")
    print(MODEL_URL)
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as exc:
        print("\nCould not download the model automatically.")
        print("Download it manually from the URL above and save it as:")
        print(MODEL_PATH)
        raise RuntimeError("Model download failed") from exc

    print(f"Saved model to: {MODEL_PATH}")


def create_landmarker():
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def result_to_dict(result, timestamp_ms):
    """Convert MediaPipe result to the same compact structure useful for Unity."""
    hands = []

    for index, landmarks in enumerate(result.hand_landmarks):
        label = "Unknown"
        score = 0.0

        if index < len(result.handedness) and result.handedness[index]:
            category = result.handedness[index][0]
            label = category.category_name or "Unknown"
            score = float(category.score)

        hands.append(
            {
                "label": label,
                "score": score,
                "landmarks": [
                    {
                        "x": float(lm.x),
                        "y": float(lm.y),
                        "z": float(lm.z),
                    }
                    for lm in landmarks
                ],
            }
        )

    return {
        "timestamp_ms": int(timestamp_ms),
        "hands": hands,
    }


def write_json_atomic(data):
    """Write JSON without leaving a half-written file for future Unity readers."""
    temp = JSON_PATH.with_suffix(".tmp")
    temp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    temp.replace(JSON_PATH)


def landmark_to_pixel(lm, width, height):
    x = int(np.clip(lm.x, 0.0, 1.0) * (width - 1))
    y = int(np.clip(lm.y, 0.0, 1.0) * (height - 1))
    return x, y


def draw_hand(frame, landmarks, label, score, color):
    height, width = frame.shape[:2]
    pts = [landmark_to_pixel(lm, width, height) for lm in landmarks]

    # Draw bones first.
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 3, cv2.LINE_AA)

    # Draw joints.
    for i, point in enumerate(pts):
        radius = 7 if i in (4, 8, 12, 16, 20) else 5
        cv2.circle(frame, point, radius, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(frame, point, max(2, radius - 2), color, -1, cv2.LINE_AA)

    # Label near wrist.
    wx, wy = pts[0]
    label_text = f"{label}  {score * 100:.0f}%"
    cv2.putText(
        frame,
        label_text,
        (max(8, wx - 45), max(28, wy - 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_panel(frame, fps, hand_count, json_enabled, show_help):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    panel_height = 104 if show_help else 70
    cv2.rectangle(overlay, (0, 0), (w, panel_height), PANEL_COLOR, -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    status = f"Hands: {hand_count}/2   FPS: {fps:4.1f}"
    cv2.putText(
        frame, status, (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.72, GOOD_COLOR, 2, cv2.LINE_AA
    )

    json_text = "JSON: ON" if json_enabled else "JSON: OFF"
    cv2.putText(
        frame, json_text, (18, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.58, TEXT_COLOR, 1, cv2.LINE_AA
    )

    if show_help:
        cv2.putText(
            frame, "Q/Esc quit   S screenshot   J JSON output   H help",
            (18, 91),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, TEXT_COLOR, 1, cv2.LINE_AA
        )


def open_camera(index, width, height):
    # CAP_AVFOUNDATION helps on macOS; fall back to generic capture if unavailable.
    if sys.platform == "darwin":
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(index)
    else:
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {index}. "
            "Try --camera 1 if you have more than one camera."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def save_screenshot(frame):
    SCREENSHOT_DIR.mkdir(exist_ok=True)
    filename = SCREENSHOT_DIR / time.strftime("hand_tracking_%Y%m%d_%H%M%S.png")
    cv2.imwrite(str(filename), frame)
    print(f"Saved screenshot: {filename}")


def parse_args():
    parser = argparse.ArgumentParser(description="Laptop MediaPipe hand tracking POC")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height")
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Do not mirror the preview. Default preview behaves like a selfie camera.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write latest_landmarks.json continuously from startup.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_model()

    cap = open_camera(args.camera, args.width, args.height)
    landmarker = create_landmarker()

    json_enabled = args.json
    show_help = True
    last_timestamp_ms = 0

    fps = 0.0
    frame_counter = 0
    fps_timer = time.perf_counter()

    print()
    print("=" * 62)
    print(" Laptop Hand Tracking Proof of Concept")
    print("=" * 62)
    print("Show one or two hands to your laptop camera.")
    print("Q / Esc = quit")
    print("S       = save screenshot")
    print("J       = toggle landmark JSON file")
    print("H       = toggle help text")
    print("=" * 62)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame read failed.")
                break

            # Mirror for a natural laptop-camera demo.
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            # MediaPipe expects RGB.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )

            timestamp_ms = int(time.monotonic() * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # Draw each hand.
            for index, landmarks in enumerate(result.hand_landmarks):
                label = "Unknown"
                score = 0.0

                if index < len(result.handedness) and result.handedness[index]:
                    category = result.handedness[index][0]
                    label = category.category_name or "Unknown"
                    score = float(category.score)

                # A mirrored preview reverses the visual side. Keeping MediaPipe's
                # handedness label unchanged is useful because this same data format
                # will later be used by the HoloLens pipeline.
                color = LEFT_COLOR if label.lower() == "left" else RIGHT_COLOR
                draw_hand(frame, landmarks, label, score, color)

            data = result_to_dict(result, timestamp_ms)
            if json_enabled:
                write_json_atomic(data)

            frame_counter += 1
            now = time.perf_counter()
            elapsed = now - fps_timer
            if elapsed >= 0.5:
                fps = frame_counter / elapsed
                frame_counter = 0
                fps_timer = now

            draw_panel(
                frame,
                fps=fps,
                hand_count=len(result.hand_landmarks),
                json_enabled=json_enabled,
                show_help=show_help,
            )

            cv2.imshow("HoloLens Project - Laptop Hand Tracking POC", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("s"), ord("S")):
                save_screenshot(frame)
            elif key in (ord("j"), ord("J")):
                json_enabled = not json_enabled
                print(f"JSON output {'enabled' if json_enabled else 'disabled'}")
            elif key in (ord("h"), ord("H")):
                show_help = not show_help

    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
