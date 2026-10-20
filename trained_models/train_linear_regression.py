from pathlib import Path
import json

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from train_model import ROOT
from train_model_v2 import prepare_v2
from train_classical import aggregate

SEED = 42


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


def one_hot(targets, num_classes):
    matrix = np.zeros((len(targets), num_classes), dtype=np.float32)
    matrix[np.arange(len(targets)), targets] = 1.0
    return matrix


def predict_labels(model, x):
    # Linear regression has no notion of class, so each of the 15
    # outputs is fit by least squares against a 0/1 indicator column,
    # and the class with the highest predicted score wins.
    scores = model.predict(x)
    return np.argmax(scores, axis=1)


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

    train_targets = one_hot(train_y, len(labels))

    print("Technique: Linear Regression (least squares, one-vs-rest indicators)")
    print("Aggregated feature vector size:", train_x.shape[-1])
    print("Training videos:", len(train_y))
    print("Validation videos:", len(val_y))
    print("Classes:", labels, flush=True)

    model = LinearRegression()
    model.fit(train_x, train_targets)

    train_predictions = predict_labels(model, train_x)
    val_predictions = predict_labels(model, val_x)

    train_accuracy = float(np.mean(train_predictions == train_y))
    val_accuracy = float(np.mean(val_predictions == val_y))

    print(f"Train accuracy: {train_accuracy:.1%}")
    print(f"Validation accuracy: {val_accuracy:.1%}")

    destination = ROOT / "trained_models/trained_model_linear_regression"
    destination.mkdir(exist_ok=True)

    joblib.dump(model, destination / "model.joblib")
    joblib.dump(scaler, destination / "scaler.joblib")

    (destination / "labels.json").write_text(
        json.dumps(labels, indent=2)
    )
    (destination / "training_results.json").write_text(
        json.dumps({
            "technique": "linear_regression",
            "train_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy
        }, indent=2)
    )

    print("\nTRAINING COMPLETE")
    print("Saved model:", destination / "model.joblib")


if __name__ == "__main__":
    main()
