from pathlib import Path
import json

import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from train_model import ROOT
from train_model_v2 import prepare_v2
from train_classical import aggregate

SEED = 42
# Kernels covered: polynomial, RBF/Gaussian, sigmoid (linear is included
# as the no-kernel-trick baseline for comparison).
CANDIDATE_KERNELS = ["linear", "poly", "rbf", "sigmoid"]
C = 10.0


def load_split(split, labels):
    paths = sorted((ROOT / "features" / split).glob("*.npz"))
    if not paths:
        raise RuntimeError(f"No feature files found for {split}")

    examples = []
    targets = []

    for path in paths:
        features, label = prepare_v2(path)
        examples.append(aggregate(features))
        targets.append(labels.index(label))

    return np.stack(examples), np.array(targets)


def main():
    training_paths = sorted((ROOT / "features/train").glob("*.npz"))
    labels = set()
    for path in training_paths:
        with np.load(path, allow_pickle=False) as data:
            labels.add(str(data["gloss"].item()))
    labels = sorted(labels)

    if len(labels) != 15:
        raise RuntimeError(f"Expected 15 labels, found {len(labels)}")

    train_x, train_y = load_split("train", labels)
    val_x, val_y = load_split("val", labels)

    if len(train_y) != 218 or len(val_y) != 57:
        raise RuntimeError("Unexpected number of training/validation files")

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    val_x = scaler.transform(val_x)

    print("Technique: Support Vector Machine (max-margin classifier)")
    print("Aggregated feature vector size:", train_x.shape[-1])
    print("Training videos:", len(train_y))
    print("Validation videos:", len(val_y))
    print("Classes:", labels, flush=True)
    print(f"C (margin/error trade-off): {C}")

    # The margin, and which points become support vectors, depends on
    # how the kernel maps the data, so sweep kernels and keep the best.
    best_model = None
    best_kernel = None
    best_val_accuracy = -1.0

    for kernel in CANDIDATE_KERNELS:
        model = SVC(kernel=kernel, C=C, gamma="scale", random_state=SEED)
        model.fit(train_x, train_y)

        val_accuracy = model.score(val_x, val_y)
        print(f"  kernel={kernel:7} validation accuracy={val_accuracy:.1%}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_kernel = kernel
            best_model = model

    train_accuracy = best_model.score(train_x, train_y)

    print(f"\nBest kernel: {best_kernel}")
    print(f"Train accuracy: {train_accuracy:.1%}")
    print(f"Validation accuracy: {best_val_accuracy:.1%}")
    print("Support vectors per class:", best_model.n_support_.tolist())

    destination = ROOT / "trained_models/trained_model_svm"
    destination.mkdir(exist_ok=True)

    joblib.dump(best_model, destination / "model.joblib")
    joblib.dump(scaler, destination / "scaler.joblib")

    (destination / "labels.json").write_text(
        json.dumps(labels, indent=2)
    )
    (destination / "training_results.json").write_text(
        json.dumps({
            "technique": "svm",
            "kernel": best_kernel,
            "C": C,
            "train_accuracy": train_accuracy,
            "validation_accuracy": best_val_accuracy
        }, indent=2)
    )

    print("\nTRAINING COMPLETE")
    print("Saved model:", destination / "model.joblib")


if __name__ == "__main__":
    main()
