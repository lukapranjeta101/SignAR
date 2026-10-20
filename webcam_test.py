"""Run beside train_model.py, train_model_v2.py and extract_landmarks.py.
SPACE: record one sign; A/D: choose intended word; Q: quit.
Records temporarily, reuses the original extractor and preprocessing, and
logs predictions to webcam_results.csv. No training data is modified.
"""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "trained_models"))

from extract_landmarks import extract_video
from train_model import SignModel, prepare_video
from train_model_v2 import prepare_v2


def load_models():
    paths = [ROOT / 'trained_models/trained_model/sign_model.pt']
    paths += sorted((ROOT / 'trained_models/trained_model_v2').glob('seed_*/sign_model.pt'))
    loaded = []
    for path in paths:
        if not path.exists():
            continue
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        kind = checkpoint.get('preprocessing')
        if kind not in ('shoulder_xy_masks_v1', 'wrist_shape_motion_v2'):
            raise ValueError(f'Unsupported preprocessing in {path}: {kind}')
        if checkpoint.get('steps') != 48:
            raise ValueError(f'Expected 48 steps: {path}')
        labels = checkpoint['labels']
        model = SignModel(checkpoint['input_size'], len(labels))
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.eval()
        loaded.append((str(path.parent.relative_to(ROOT)), model, labels,
                       kind, checkpoint['input_size']))
    if not loaded:
        raise RuntimeError('No saved models found. Put this script in SignAR-ML.')
    if any(item[2] != loaded[0][2] for item in loaded):
        raise ValueError('The saved models have different label lists.')
    return loaded


def predict(frames, elapsed, loaded, expected):
    # The temporary video remains unmirrored, like the original extraction input.
    with tempfile.TemporaryDirectory(prefix='signar_webcam_') as temp:
        video = Path(temp) / 'clip.avi'
        height, width = frames[0].shape[:2]
        fps = max(1.0, (len(frames) - 1) / max(elapsed, 0.001))
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'),
                                 fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError('Could not create temporary webcam video.')
        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()
        arrays = extract_video(video)
        feature_path = Path(temp) / 'clip.npz'
        np.savez_compressed(feature_path, **arrays, gloss=expected)
        prepared = {}
        rows = []
        stamp = datetime.now(timezone.utc).isoformat()
        for name, model, labels, kind, size in loaded:
            if kind not in prepared:
                fn = prepare_video if kind == 'shoulder_xy_masks_v1' else prepare_v2
                prepared[kind] = fn(feature_path)[0]
            features = prepared[kind]
            if features.shape != (48, size):
                raise ValueError(f'{name}: unexpected input shape {features.shape}')
            with torch.inference_mode():
                scores = torch.softmax(model(torch.from_numpy(features).unsqueeze(0)), 1)[0]
            values, indices = torch.topk(scores, min(3, len(labels)))
            word = labels[int(indices[0])]
            top3 = '; '.join(f'{labels[int(i)]}: {float(v):.1%}'
                             for v, i in zip(values, indices))
            rows.append(dict(timestamp=stamp, model=name, expected=expected,
                             predicted=word, correct=int(word == expected),
                             model_score=float(values[0]), top3=top3,
                             hand_detection_rate=float(arrays['hand_present'].any(axis=1).mean())))
        return rows


def draw(frame, lines):
    # Mirror only the preview. Inference always sees the raw camera frames.
    preview = cv2.flip(frame, 1)
    panel_height = min(preview.shape[0], 35 + len(lines) * 25)
    cv2.rectangle(preview, (0, 0), (preview.shape[1], panel_height), (20, 20, 20), -1)
    for index, line in enumerate(lines):
        cv2.putText(preview, line, (12, 28 + index * 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.imshow('SignAR webcam test', preview)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--seconds', type=float, default=3.0)
    args = parser.parse_args()
    if not 0.5 <= args.seconds <= 10:
        parser.error('--seconds must be between 0.5 and 10')
    loaded = load_models()
    print('Loaded models:', *(item[0] for item in loaded), sep='\n  ')
    print('Model scores are not calibrated probabilities of correctness.')
    print('Choose the intended word BEFORE recording. SPACE starts a countdown.')
    labels = loaded[0][2]
    selected = 0
    results = ['Show your face, shoulders, elbows and both hands.']
    log = ROOT / 'webcam_results.csv'
    capture = cv2.VideoCapture(args.camera)
    try:
        if not capture.isOpened():
            raise RuntimeError('Camera unavailable. Allow camera access for VS Code/Terminal, or try --camera 1.')
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        state = 'idle'
        frames = []
        deadline = 0.0
        first_time = last_time = 0.0
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError('Camera stopped returning frames.')
            now = time.monotonic()
            if state == 'countdown' and now >= deadline:
                state = 'recording'
                frames = []
                first_time = now
                deadline = now + args.seconds
            if state == 'recording':
                frames.append(frame.copy())
                last_time = now
                if now >= deadline:
                    draw(frame, ['Processing recorded sign...'])
                    cv2.waitKey(1)
                    try:
                        if len(frames) < 10:
                            raise ValueError('Too few frames. Try recording again.')
                        rows = predict(frames, last_time - first_time, loaded, labels[selected])
                        results = []
                        for row in rows:
                            mark = 'MATCH' if row['correct'] else 'MISS'
                            results.append(f"{row['model']}: {row['predicted']} {row['model_score']:.0%} {mark}")
                            print(f"{row['model']} | intended {row['expected']} | {row['top3']}")
                        results.append(f"Hands detected: {rows[0]['hand_detection_rate']:.0%} of frames")
                        has_header = log.exists() and log.stat().st_size > 0
                        with log.open('a', newline='') as output:
                            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                            if not has_header:
                                writer.writeheader()
                            writer.writerows(rows)
                        print('Saved results:', log)
                    except Exception as error:
                        results = ['Recording could not be evaluated. See terminal.']
                        print('Evaluation error:', error)
                    frames = []
                    state = 'idle'
            lines = [f'Intended word: {labels[selected]}', 'A/D: word | SPACE: record | Q: quit']
            if state == 'countdown':
                lines += [f'Get ready: {max(1, int(deadline - now) + 1)}']
            elif state == 'recording':
                lines += [f'SIGN NOW - {max(0, deadline - now):.1f}s remaining']
            else:
                lines += results
            draw(frame, lines)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if cv2.getWindowProperty('SignAR webcam test', cv2.WND_PROP_VISIBLE) < 1:
                break
            if state == 'idle':
                if key == ord('a'):
                    selected = (selected - 1) % len(labels)
                    results = []
                elif key == ord('d'):
                    selected = (selected + 1) % len(labels)
                    results = []
                elif key == 32:
                    state = 'countdown'
                    deadline = time.monotonic() + 2
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
