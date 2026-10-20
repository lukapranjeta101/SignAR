from pathlib import Path
from collections import Counter
import json

import numpy as np

from train_model import ROOT
from train_model_v2 import prepare_v2

# DTW cost grows with sequence length squared, so the 48-step sequences
# are downsampled further for this technique (a lazy, distance-based
# classifier with no gradient-based training).
DTW_STEPS = 16
CANDIDATE_K = [1, 3, 5, 7, 9]


def prepare_trajectory(path):
    features, label = prepare_v2(path)
    # Keep only the raw hand + body xy coordinates (the first 106 of the
    # v2 feature layout), dropping visibility masks, shape and motion
    # features that the other techniques use — DTW compares trajectories
    # of the keypoints directly.
    coordinates = features[:, :106]

    indices = np.rint(
        np.linspace(0, len(coordinates) - 1, DTW_STEPS)
    ).astype(int)

    return coordinates[indices].astype(np.float32), label


def load_split(split, labels):
    paths = sorted((ROOT / "features" / split).glob("*.npz"))
    if not paths:
        raise RuntimeError(f"No feature files found for {split}")

    examples = []
    targets = []

    for path in paths:
        trajectory, label = prepare_trajectory(path)
        examples.append(trajectory)
        targets.append(labels.index(label))

    return np.stack(examples), np.array(targets)


def dtw_distance(a, b):
    # a, b: (DTW_STEPS, features). Classic DTW dynamic program over a
    # per-frame squared Euclidean cost matrix.
    cost = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
    steps_a, steps_b = cost.shape

    table = np.full((steps_a + 1, steps_b + 1), np.inf)
    table[0, 0] = 0.0

    for i in range(1, steps_a + 1):
        for j in range(1, steps_b + 1):
            table[i, j] = cost[i - 1, j - 1] + min(
                table[i - 1, j],
                table[i, j - 1],
                table[i - 1, j - 1]
            )

    return table[steps_a, steps_b]


def classify(query, train_x, train_y, k):
    distances = np.array(
        [dtw_distance(query, sequence) for sequence in train_x]
    )
    return classify_from_distances(distances, train_y, k)


def classify_from_distances(distances, train_y, k):
    nearest = np.argsort(distances)[:k]
    votes = Counter(train_y[index] for index in nearest)
    return votes.most_common(1)[0][0]


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

    print("Technique: k-NN with Dynamic Time Warping")
    print("Trajectory steps:", DTW_STEPS)
    print("Training videos:", len(train_y))
    print("Validation videos:", len(val_y))
    print("Classes:", labels, flush=True)

    print("Computing DTW distances between validation and training clips...")
    distance_matrix = np.array([
        [dtw_distance(query, sequence) for sequence in train_x]
        for query in val_x
    ])

    best_k = None
    best_accuracy = -1.0

    for k in CANDIDATE_K:
        predictions = np.array([
            classify_from_distances(row, train_y, k)
            for row in distance_matrix
        ])
        accuracy = float((predictions == val_y).mean())
        print(f"k={k:2d} | validation accuracy {accuracy:.1%}", flush=True)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k

    print(f"\nSelected k={best_k} (validation accuracy {best_accuracy:.1%})")

    destination = ROOT / "trained_models/trained_model_dtw"
    destination.mkdir(exist_ok=True)

    np.savez(
        destination / "model.npz",
        train_x=train_x,
        train_y=train_y
    )
    (destination / "labels.json").write_text(json.dumps(labels, indent=2))
    (destination / "training_results.json").write_text(
        json.dumps({
            "technique": "dtw_knn",
            "dtw_steps": DTW_STEPS,
            "k": best_k,
            "validation_accuracy": best_accuracy
        }, indent=2)
    )

    print("\nTRAINING COMPLETE")
    print("Saved model:", destination / "model.npz")


if __name__ == "__main__":
    main()
