# SignAR-ML — Windows Setup Guide

This project was developed on macOS and is being moved to Windows. Follow these
steps in order to get the environment, data, and models running again.

## 0. What to copy over

Copy the **entire** `SignAR-ML` folder to the Windows machine, with one exception:

- **Do not copy `.venv/`** (and you can skip `__pycache__/` too). It's a macOS
  virtual environment and will not work on Windows — you'll recreate it in
  Step 2.

Everything else — `data/`, `features/`, `models/`, `trained_model/`,
`trained_model_v2/`, and all the `.py` scripts — must come along as plain files.
These folders are listed in `.gitignore`, so `git clone` alone will **not**
bring them over; you need to physically copy the folder (USB drive, external
SSD, cloud sync, etc.), not just copy the git repo.

Rough sizes, so you know what you're moving:

| Folder | Size | Contents |
|---|---|---|
| `data/` | ~240 MB | ASL Citizen source videos |
| `features/` | ~22 MB | extracted landmark `.npz` features (train/val/test) |
| `models/` | ~16 MB | MediaPipe `hand_landmarker.task` / `pose_landmarker.task` |
| `trained_model/`, `trained_model_v2/` | ~2 MB | trained checkpoints |

## 1. Install prerequisites on Windows

- **Python 3.14** (64-bit) — [python.org/downloads](https://www.python.org/downloads/).
  During install, check **"Add python.exe to PATH"**.
  - All pinned packages in `requirements-working.txt` (mediapipe, torch,
    opencv-contrib-python, scipy, pandas, scikit-learn, etc.) publish official
    `win_amd64` wheels for 3.14, so you don't need to downgrade Python to make
    this work.
- **Git for Windows** (optional, only needed if you'll keep committing code) —
  [git-scm.com/download/win](https://git-scm.com/download/win).
- A webcam, if you plan to run `webcam_test.py`.

Verify the install by opening **PowerShell** and running:

```powershell
python --version
```

## 2. Create the virtual environment and install dependencies

From inside the project folder in PowerShell:

```powershell
cd path\to\SignAR-ML
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If activation is blocked by PowerShell's execution policy, run this once
(as the current user, not admin) and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

With the venv active (`(.venv)` shows in the prompt), install the pinned
dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements-working.txt
```

This installs CPU-only PyTorch, MediaPipe, OpenCV, scikit-learn, etc. at the
exact versions used during development, so training/evaluation results should
match the originals.

## 3. Sanity-check the data and models arrived intact

```powershell
dir data\ASL_Citizen\videos
dir features\train
dir models
dir trained_model
```

You should see `.mp4` files under `data`, `.npz` files under `features`,
`hand_landmarker.task` / `pose_landmarker.task` under `models`, and
`sign_model.pt` / `labels.json` under `trained_model`.

If `features/` or `models/` came over empty, re-copy them before continuing —
training and evaluation scripts read from these paths directly and will fail
fast with a clear error if files are missing.

## 4. (Optional) Re-extract features from raw video

Only needed if `features/` didn't transfer or you add new videos to
`data/ASL_Citizen/videos`:

```powershell
python extract_landmarks.py
```

This reads `models/hand_landmarker.task` and `models/pose_landmarker.task` via
MediaPipe and writes `.npz` feature files into `features/train`, `features/val`,
`features/test`.

## 5. Train a model

Each technique is a standalone script and writes its own output folder:

```powershell
python train_classical.py     # -> trained_model_classical/   (Random Forest)
python train_dtw_knn.py       # -> trained_model_dtw/         (DTW + k-NN)
python train_lstm.py          # -> trained_model_lstm/        (BiLSTM)
python train_transformer.py   # -> trained_model_transformer/ (Transformer)
```

Each prints train/validation accuracy as it runs and saves the model, labels,
and training metadata into its output folder.

The original `train_model.py` / `train_model_v2.py` scripts produce
`trained_model/` and `trained_model_v2/` (already present if those folders
transferred over) and are what `webcam_test.py` and `prediction_server.py`
load by default.

## 6. Evaluate a trained model on the held-out test set

```powershell
python evaluate_classical.py
python evaluate_dtw.py
python evaluate_lstm.py
python evaluate_transformer.py
```

Each expects exactly 190 files in `features/test` and writes
`test_predictions.csv` and `test_confusion_matrix.csv` into its model's output
folder.

## 7. Run the live webcam demo

```powershell
python webcam_test.py
```

- Windows will prompt for camera permission the first time — allow access for
  your terminal/VS Code.
- If the camera fails to open, try a different index: `python webcam_test.py --camera 1`.
- Controls: **SPACE** to record a sign, **A/D** to pick the intended word,
  **Q** to quit. Predictions are logged to `webcam_results.csv`.

## 8. Run the prediction server demo

```powershell
python prediction_server.py
```

Loads `trained_model/sign_model.pt` and runs a one-off prediction against a
sample "HELLO" validation clip to confirm the model loads and infers correctly.

## Troubleshooting

- **`ModuleNotFoundError` for any package** — make sure `(.venv)` is active in
  your terminal (`.venv\Scripts\Activate.ps1`), then re-run `pip install -r requirements-working.txt`.
- **`RuntimeError: Camera unavailable`** — close any other app using the
  webcam (Zoom, Teams, etc.) and check Windows Settings → Privacy & security →
  Camera → allow desktop apps.
- **Paths with backslashes** — all scripts use `pathlib.Path`, so they work
  identically on Windows; you never need to edit paths by hand.
- **Antivirus/SmartScreen flags `.task` model files or `.pt` checkpoints** —
  these are legitimate MediaPipe/PyTorch model files; allow them if prompted.
