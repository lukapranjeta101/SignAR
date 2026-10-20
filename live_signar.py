"""Live rolling-window SignAR prototype. Run beside the existing training scripts.
No video/feature files are written. Last accepted word stays visible until replaced. R clears history and word; Q exits.
Scores and temporal agreement are heuristics, not an idle/unknown classifier.
"""
import argparse
from collections import deque
from contextlib import ExitStack
from io import BytesIO
from pathlib import Path
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "trained_models"))

from train_model import SignModel, prepare_video

class MemoryClip(BytesIO):
    # Existing preprocessing uses path.name when reporting invalid inputs.
    name = 'live_window.npz'


def load_models():
    paths = [ROOT / 'trained_models/trained_model/sign_model.pt']
    paths += sorted((ROOT / 'trained_models/trained_model_v2').glob('seed_*/sign_model.pt'))
    loaded = []
    for path in paths:
        if not path.exists():
            continue
        cp = torch.load(path, map_location='cpu', weights_only=True)
        kind = cp.get('preprocessing')
        if kind not in ('shoulder_xy_masks_v1', 'wrist_shape_motion_v2'):
            raise ValueError(f'Unsupported preprocessing: {path}: {kind}')
        if cp.get('steps') != 48:
            raise ValueError(f'Expected 48 steps: {path}')
        model = SignModel(cp['input_size'], len(cp['labels']))
        model.load_state_dict(cp['state_dict'], strict=True)
        model.eval()
        loaded.append((model, cp['labels'], kind, cp['input_size']))
        print('Loaded:', path.relative_to(ROOT))
    if not loaded:
        raise RuntimeError('No saved models found. Put this file in SignAR-ML.')
    if any(item[1] != loaded[0][1] for item in loaded):
        raise ValueError('Models have different label lists.')
    return loaded


def extract_frame(frame, timestamp, hands, pose):
    image = mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    hr = hands.detect_for_video(image, timestamp)
    pr = pose.detect_for_video(image, timestamp)
    hp = np.zeros((2, 21, 3), np.float32)
    hm = np.zeros(2, bool)
    best = [-1., -1.]
    for landmarks, categories in zip(hr.hand_landmarks, hr.handedness):
        if not categories or categories[0].category_name not in ('Left', 'Right'):
            continue
        category = categories[0]
        slot = 0 if category.category_name == 'Left' else 1
        if category.score <= best[slot]:
            continue
        best[slot] = category.score
        hp[slot] = [[p.x, p.y, p.z] for p in landmarks]
        hm[slot] = True
    pp = np.zeros((33, 4), np.float32)
    pm = bool(pr.pose_landmarks)
    if pm:
        pp[:] = [[p.x, p.y, p.z, p.visibility] for p in pr.pose_landmarks[0]]
    return hp, hm, pp, pm


def prepare_v2_memory(clip):
    """Same V2 features as training; rewind before each NPZ read."""
    clip.seek(0)
    baseline, label = prepare_video(clip)
    clip.seek(0)
    with np.load(clip, allow_pickle=False) as data:
        hands = data['hands'].copy()
        masks = data['hand_present'].copy()
    active = np.flatnonzero(masks.any(axis=1))
    if len(active) == 0:
        raise ValueError('No detected hands')
    start = max(0, int(active[0]) - 5)
    end = min(len(hands), int(active[-1]) + 6)
    hands, masks = hands[start:end], masks[start:end]
    indices = np.rint(np.linspace(0, len(hands) - 1, 48)).astype(int)
    hands, masks = hands[indices], masks[indices]
    xy = hands[:, :, :, :2]
    relative = xy - xy[:, :, 0:1, :]
    palm_size = np.linalg.norm(relative[:, :, 9, :], axis=-1)
    shape_valid = masks & (palm_size > 0.005)
    shape = relative / np.maximum(palm_size[:, :, None, None], 0.005)
    shape[~shape_valid] = 0
    coordinates = baseline[:, :106]
    hand_present = baseline[:, 106:108] > 0.5
    body_present = baseline[:, 108:119] > 0.5
    visible = np.concatenate([np.repeat(hand_present, 42, axis=1),
                              np.repeat(body_present, 2, axis=1)], axis=1)
    movement = np.zeros_like(coordinates)
    movement_valid = np.zeros_like(visible)
    movement_valid[1:] = visible[1:] & visible[:-1]
    movement[1:] = coordinates[1:] - coordinates[:-1]
    movement[~movement_valid] = 0
    features = np.concatenate([baseline, shape.reshape(48, -1),
                               shape_valid.astype(np.float32), movement,
                               movement_valid.astype(np.float32)], axis=1).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError('Invalid live features')
    return features, label


def predict(window, loaded):
    # In-memory NPZ preserves the exact original preprocessing without disk I/O.
    clip = MemoryClip()
    np.savez(clip, hands=np.stack([x[1] for x in window]),
             hand_present=np.stack([x[2] for x in window]),
             pose=np.stack([x[3] for x in window]),
             pose_present=np.asarray([x[4] for x in window], bool),
             gloss=np.asarray('LIVE'))
    prepared = {}
    scores = []
    with clip, torch.inference_mode():
        for model, labels, kind, size in loaded:
            if kind not in prepared:
                clip.seek(0)
                fn = prepare_video if kind == 'shoulder_xy_masks_v1' else prepare_v2_memory
                prepared[kind] = fn(clip)[0]
            features = prepared[kind]
            if features.shape != (48, size):
                raise ValueError(f'Unexpected feature shape: {features.shape}')
            logits = model(torch.from_numpy(features).unsqueeze(0))
            scores.append(torch.softmax(logits, dim=1)[0])
    # Equal-weight average of baseline and all available V2 model scores.
    mean = torch.stack(scores).mean(dim=0)
    values, indices = torch.topk(mean, min(3, len(mean)))
    return labels[int(indices[0])], float(values[0]), ' | '.join(
        f'{labels[int(i)]} {float(v):.0%}' for v, i in zip(values, indices))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--window', type=float, default=2.0)
    parser.add_argument('--interval', type=float, default=0.25)
    parser.add_argument('--threshold', type=float, default=0.75)
    args = parser.parse_args()
    if not 0.5 <= args.window <= 10 or args.interval < 0.05 or not 0 <= args.threshold <= 1:
        parser.error('window: 0.5–10; interval: >=0.05; threshold: 0–1')
    loaded = load_models()
    window = deque(maxlen=1000)
    history = deque(maxlen=3)
    status, top3 = 'Warming up...', ''
    displayed_word = None
    displayed_score = 0.0
    started = time.monotonic()
    previous_timestamp = -1
    last_prediction = 0.
    print('Click the camera window. Q: quit; R: clear rolling window.')
    print('Scores are not accuracy. This model has no idle/unknown training class.')
    with ExitStack() as stack:
        hands = stack.enter_context(vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(ROOT / 'models/hand_landmarker.task'),
                                                delegate=python.BaseOptions.Delegate.CPU),
                running_mode=vision.RunningMode.VIDEO, num_hands=2)))
        pose = stack.enter_context(vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(ROOT / 'models/pose_landmarker.task'),
                                                delegate=python.BaseOptions.Delegate.CPU),
                running_mode=vision.RunningMode.VIDEO, num_poses=1)))
        capture = cv2.VideoCapture(args.camera)
        stack.callback(cv2.destroyAllWindows)
        stack.callback(capture.release)
        if not capture.isOpened():
            raise RuntimeError('Camera unavailable. Allow camera access or try --camera 1.')
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError('Camera stopped returning frames.')
            now = time.monotonic()
            timestamp = max(previous_timestamp + 1, round((now - started) * 1000))
            previous_timestamp = timestamp
            hp, hm, pp, pm = extract_frame(frame, timestamp, hands, pose)
            window.append((now, hp, hm, pp, pm))
            while window and window[0][0] < now - args.window:
                window.popleft()
            if not hm.any():
                status, top3 = 'No hands visible', ''
                history.clear()
            elif len(window) < 10 or now - window[0][0] < args.window * 0.8:
                status = 'Warming up... Keep shoulders and hands visible.'
                top3 = ''
            elif now - last_prediction >= args.interval:
                last_prediction = now
                try:
                    word, score, top3 = predict(window, loaded)
                    if score >= args.threshold:
                        history.append(word)
                    else:
                        history.clear()
                    if len(history) == 3 and len(set(history)) == 1:
                        displayed_word, displayed_score = word, score
                        status = 'Consistent prediction accepted'
                    else:
                        status = 'Uncertain / waiting for consistent predictions'
                except ValueError as error:
                    status, top3 = str(error), ''
                    history.clear()
            preview = cv2.flip(frame, 1)  # Inference receives unmirrored frames.
            headline = (f'Last prediction: {displayed_word} | score {displayed_score:.0%}'
                        if displayed_word is not None else 'Last prediction: waiting for a sign')
            lines = [headline, status, top3, 'Sign one word at a time. Q: quit | R: reset',
                     f'Window: {args.window:.1f}s | frames: {len(window)}']
            cv2.rectangle(preview, (0, 0), (preview.shape[1], 150), (20, 20, 20), -1)
            for i, line in enumerate(lines):
                cv2.putText(preview, line, (10, 25 + i * 27), cv2.FONT_HERSHEY_SIMPLEX,
                            0.48, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.imshow('SignAR live', preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27) or cv2.getWindowProperty('SignAR live', cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord('r'):
                displayed_word, displayed_score = None, 0.0
                window.clear()
                history.clear()
                status, top3 = 'Warming up...', ''


if __name__ == '__main__':
    main()
