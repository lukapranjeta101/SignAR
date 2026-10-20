from pathlib import Path
import json

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
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

    print("Technique: Multilayer Perceptron (hidden layers, non-linear boundary)")
    print("Aggregated feature vector size:", train_x.shape[-1])
    print("Training videos:", len(train_y))
    print("Validation videos:", len(val_y))
    print("Classes:", labels, flush=True)

    # Two hidden layers let the network combine the perceptron's linear
    # units into a non-linear decision boundary, unlike a single layer.
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=1e-3,
        max_iter=2000,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.15,
        random_state=SEED
    )
    model.fit(train_x, train_y)

    train_accuracy = model.score(train_x, train_y)
    val_accuracy = model.score(val_x, val_y)

    print(f"Train accuracy: {train_accuracy:.1%}")
    print(f"Validation accuracy: {val_accuracy:.1%}")
    print(f"Training stopped after {model.n_iter_} iterations")

    destination = ROOT / "trained_models/trained_model_mlp"
    destination.mkdir(exist_ok=True)

    joblib.dump(model, destination / "model.joblib")
    joblib.dump(scaler, destination / "scaler.joblib")

    (destination / "labels.json").write_text(
        json.dumps(labels, indent=2)
    )
    (destination / "training_results.json").write_text(
        json.dumps({
            "technique": "mlp",
            "hidden_layer_sizes": [64, 32],
            "train_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
            "iterations": model.n_iter_
        }, indent=2)
    )

    print("\nTRAINING COMPLETE")
    print("Saved model:", destination / "model.joblib")


if __name__ == "__main__":
    main()
