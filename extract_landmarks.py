from pathlib import Path
import csv
import math

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "ASL_Citizen"
OUTPUT = ROOT / "features"
OUTPUT.mkdir(exist_ok=True)


def extract_video(video_path):
    capture = cv2.VideoCapture(str(video_path))

    try:
        if not capture.isOpened():
            raise RuntimeError("Could not open video")

        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise RuntimeError("Invalid video frame rate")

        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(ROOT / "models/hand_landmarker.task"),
                delegate=python.BaseOptions.Delegate.CPU
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2
        )

        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(ROOT / "models/pose_landmarker.task"),
                delegate=python.BaseOptions.Delegate.CPU
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1
        )

        hand_frames = []
        hand_masks = []
        pose_frames = []
        pose_masks = []
        timestamps = []
        previous_time = -1

        # New trackers for each video prevent tracking across different people.
        with vision.HandLandmarker.create_from_options(hand_options) as hands:
            with vision.PoseLandmarker.create_from_options(pose_options) as pose:
                while True:
                    success, frame = capture.read()
                    if not success:
                        break

                    timestamp = max(
                        previous_time + 1,
                        round(len(timestamps) * 1000 / fps)
                    )
                    previous_time = timestamp

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb
                    )

                    hand_result = hands.detect_for_video(image, timestamp)
                    pose_result = pose.detect_for_video(image, timestamp)

                    # Slots use MediaPipe's handedness: 0 = Left, 1 = Right.
                    # Each hand has 21 points with x, y, z coordinates.
                    hand_points = np.zeros((2, 21, 3), dtype=np.float32)
                    hand_present = np.zeros(2, dtype=bool)
                    best_scores = [-1.0, -1.0]

                    for landmarks, categories in zip(
                        hand_result.hand_landmarks,
                        hand_result.handedness
                    ):
                        if not categories:
                            continue

                        category = categories[0]
                        label = category.category_name

                        if label not in ("Left", "Right"):
                            continue

                        slot = 0 if label == "Left" else 1

                        # If two detections get the same label, keep the
                        # higher-confidence detection in that slot.
                        if category.score <= best_scores[slot]:
                            continue

                        best_scores[slot] = category.score
                        hand_points[slot] = [
                            [point.x, point.y, point.z]
                            for point in landmarks
                        ]
                        hand_present[slot] = True

                    # Body: 33 points with x, y, z and visibility.
                    pose_points = np.zeros((33, 4), dtype=np.float32)
                    pose_present = bool(pose_result.pose_landmarks)

                    if pose_present:
                        pose_points[:] = [
                            [point.x, point.y, point.z, point.visibility]
                            for point in pose_result.pose_landmarks[0]
                        ]

                    hand_frames.append(hand_points)
                    hand_masks.append(hand_present)
                    pose_frames.append(pose_points)
                    pose_masks.append(pose_present)
                    timestamps.append(timestamp)

        if not timestamps:
            raise RuntimeError("No video frames decoded")

        return {
            "hands": np.asarray(hand_frames),
            "hand_present": np.asarray(hand_masks),
            "pose": np.asarray(pose_frames),
            "pose_present": np.asarray(pose_masks, dtype=bool),
            "timestamps_ms": np.asarray(timestamps, dtype=np.int64),
            "fps": np.asarray(fps)
        }

    finally:
        capture.release()


def main():
    tables = {
        split: pd.read_csv(DATA / "splits" / f"{split}.csv")
        for split in ("train", "val", "test")
    }

    # Confirm that the same participant is not in different splits.
    participants = {
        split: set(table["Participant ID"].astype(str))
        for split, table in tables.items()
    }

    for first, second in [
        ("train", "val"), ("train", "test"), ("val", "test")
    ]:
        overlap = participants[first] & participants[second]
        if overlap:
            raise RuntimeError(
                f"Participant overlap between {first} and {second}: {overlap}"
            )

    total = sum(len(table) for table in tables.values())
    completed = 0
    failures = 0
    report = []

    for split, table in tables.items():
        folder = OUTPUT / split
        folder.mkdir(exist_ok=True)

        for _, row in table.iterrows():
            completed += 1
            filename = str(row["Video file"])
            destination = folder / (Path(filename).stem + ".npz")
            temporary = destination.with_suffix(".tmp")

            record = {
                "split": split,
                "video": filename,
                "gloss": row["Gloss"],
                "participant": row["Participant ID"]
            }

            try:
                if not destination.exists():
                    arrays = extract_video(DATA / "videos" / filename)

                    with temporary.open("wb") as file:
                        np.savez_compressed(
                            file,
                            **arrays,
                            gloss=str(row["Gloss"]),
                            participant=str(row["Participant ID"]),
                            video_file=filename,
                            split=split
                        )

                    temporary.replace(destination)

                with np.load(destination, allow_pickle=False) as saved:
                    frames = len(saved["timestamps_ms"])
                    hand_rate = saved["hand_present"].any(axis=1).mean()
                    pose_rate = saved["pose_present"].mean()

                record.update(
                    status="ok",
                    frames=frames,
                    hand_detection_rate=float(hand_rate),
                    body_detection_rate=float(pose_rate),
                    error=""
                )

                print(
                    f"[{completed}/{total}] {split} | {row['Gloss']} | "
                    f"hands {hand_rate:.0%} | body {pose_rate:.0%}",
                    flush=True
                )

            except Exception as error:
                temporary.unlink(missing_ok=True)
                failures += 1
                record.update(status="failed", error=str(error))
                print(
                    f"[{completed}/{total}] FAILED: {filename}: {error}",
                    flush=True
                )

            report.append(record)

    report_path = OUTPUT / "extraction_report.csv"
    fields = [
        "split", "video", "gloss", "participant", "status", "frames",
        "hand_detection_rate", "body_detection_rate", "error"
    ]

    with report_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report)

    print("\nEXTRACTION FINISHED")
    print("Successful:", total - failures)
    print("Failed:", failures)
    print("Report:", report_path)


if __name__ == "__main__":
    main()