# Connect the hand tracker to SignAR

Use your existing working SignAR-ML environment. No retraining is needed.

1. Quit other camera apps and webcam scripts.
2. Copy `signar_hand_tracker.py` AND `hand_tracker.py` from this package into your SignAR-ML folder, beside `train_model.py`.
3. Keep your existing `models/hand_landmarker.task`, `models/pose_landmarker.task`, `trained_model/sign_model.pt`, and `trained_model_v2/seed_*/sign_model.pt` in that folder. This package does not contain your classifier checkpoints or pose model.
4. Open a terminal in SignAR-ML and run:

```bash
source .venv/bin/activate
python signar_hand_tracker.py --window 3 --threshold 0.5
```

On Windows, activate with `.venv\Scripts\activate` instead.

Do not run the old POC launchers or install its pinned requirements into your working SignAR environment. Those launchers start hand-only tracking, not classification.

This entry point adds body pose landmarks, uses the baseline preprocessing from `train_model.py`, and reproduces your supplied V2 feature calculation in memory. It averages baseline and available V2 model scores. Only the preview is mirrored; model inputs remain in the original camera coordinate system.

The last accepted prediction stays visible until replaced by three consecutive above-threshold predictions. R clears the window and displayed word. S saves a screenshot. J toggles `latest_landmarks.json`; its hands are now explicitly in unmirrored camera coordinates, and it includes a prediction object. A prediction can be old while hands are absent: inspect status and timestamp when consuming it. H hides help. Q quits.

Use one complete sign at a time and keep shoulders, elbows and hands visible. Threshold 0.5 is a troubleshooting starting point, not validated calibration. Idle movements and transitions can still produce false words because the model lacks an idle/unknown class. Continuous sentence recognition and HoloLens camera/network integration are not included.

Validation: Python syntax and in-memory V2 regression checked; hardware recognition requires your actual checkpoints and webcam.
