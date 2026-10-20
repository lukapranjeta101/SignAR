from pathlib import Path
import argparse
import json
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_model import SignModel, prepare_video


ROOT = Path(__file__).resolve().parent.parent
STEPS = 48


def prepare_v2(path):
    # Keep the original baseline features.
    baseline, label = prepare_video(path)

    with np.load(path, allow_pickle=False) as data:
        hands = data["hands"].copy()
        masks = data["hand_present"].copy()

    # Match the baseline's cropping and frame selection exactly.
    active = np.flatnonzero(masks.any(axis=1))
    if len(active) == 0:
        raise ValueError(f"No detected hands: {path.name}")

    start = max(0, int(active[0]) - 5)
    end = min(len(hands), int(active[-1]) + 6)

    hands = hands[start:end]
    masks = masks[start:end]

    indices = np.rint(
        np.linspace(0, len(hands) - 1, STEPS)
    ).astype(int)

    hands = hands[indices]
    masks = masks[indices]

    # Describe finger shape relative to the wrist.
    # Use x/y only, as in our baseline.
    xy = hands[:, :, :, :2]
    relative = xy - xy[:, :, 0:1, :]

    # Wrist-to-middle-finger-knuckle distance estimates palm size.
    palm_size = np.linalg.norm(relative[:, :, 9, :], axis=-1)
    shape_valid = masks & (palm_size > 0.005)

    shape = relative / np.maximum(
        palm_size[:, :, None, None], 0.005
    )
    shape[~shape_valid] = 0
    shape = shape.reshape(STEPS, -1)

    # Baseline layout:
    # 84 hand coordinates + 22 body coordinates + 13 masks.
    coordinates = baseline[:, :106]
    hand_present = baseline[:, 106:108] > 0.5
    body_present = baseline[:, 108:119] > 0.5

    visible = np.concatenate(
        [
            np.repeat(hand_present, 42, axis=1),
            np.repeat(body_present, 2, axis=1)
        ],
        axis=1
    )

    # Movement per sampled step, not physical velocity.
    # Only compute it when a point exists in both frames.
    movement = np.zeros_like(coordinates)
    movement_valid = np.zeros_like(visible)
    movement_valid[1:] = visible[1:] & visible[:-1]

    movement[1:] = coordinates[1:] - coordinates[:-1]
    movement[~movement_valid] = 0

    features = np.concatenate(
        [
            baseline,
            shape,
            shape_valid.astype(np.float32),
            movement,
            movement_valid.astype(np.float32)
        ],
        axis=1
    ).astype(np.float32)

    if not np.isfinite(features).all():
        raise ValueError(f"Invalid features: {path.name}")

    return features, label


def load_split(split, labels):
    paths = sorted((ROOT / "features" / split).glob("*.npz"))
    if not paths:
        raise RuntimeError(f"No files found for {split}")

    examples = []
    targets = []

    for path in paths:
        features, label = prepare_v2(path)
        examples.append(features)
        targets.append(labels.index(label))

    return (
        torch.from_numpy(np.stack(examples)),
        torch.tensor(targets, dtype=torch.long)
    )


def main(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Read labels from the existing model without modifying it.
    checkpoint = torch.load(
        ROOT / "trained_models/trained_model/sign_model.pt",
        map_location="cpu",
        weights_only=True
    )
    labels = checkpoint["labels"]

    train_x, train_y = load_split("train", labels)
    val_x, val_y = load_split("val", labels)

    if len(train_y) != 218 or len(val_y) != 57:
        raise RuntimeError("Unexpected number of feature files")

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("Device:", device)
    print("Seed:", seed)
    print("Features per frame:", train_x.shape[-1])
    print("Training clips:", len(train_y))
    print("Validation clips:", len(val_y))
    print("Test clips will not be loaded.", flush=True)

    # Same architecture as the baseline, with a larger input.
    model = SignModel(train_x.shape[-1], len(labels)).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.001
    )
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=16,
        shuffle=True,
        num_workers=0
    )

    val_x = val_x.to(device)
    val_y = val_y.to(device)

    output = ROOT / "trained_models/trained_model_v2" / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    best_accuracy = 0.0
    best_epoch = 0
    stale_epochs = 0
    history = []

    for epoch in range(1, 151):
        model.train()
        correct = 0
        seen = 0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            correct += (logits.argmax(dim=1) == y).sum().item()
            seen += len(y)

        model.eval()
        with torch.no_grad():
            logits = model(val_x)
            val_loss = criterion(logits, val_y).item()
            val_accuracy = (
                (logits.argmax(dim=1) == val_y)
                .float().mean().item()
            )

        if not np.isfinite(val_loss):
            raise RuntimeError("Validation loss became invalid")

        history.append({
            "epoch": epoch,
            "train_accuracy": correct / seen,
            "val_accuracy": val_accuracy,
            "val_loss": val_loss
        })

        print(
            f"Epoch {epoch:3d} | "
            f"train {correct / seen:.1%} | "
            f"validation {val_accuracy:.1%} | "
            f"val loss {val_loss:.4f}",
            flush=True
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_accuracy = val_accuracy
            best_epoch = epoch
            stale_epochs = 0

            state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

            torch.save(
                {
                    "state_dict": state,
                    "labels": labels,
                    "input_size": train_x.shape[-1],
                    "steps": STEPS,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "preprocessing": "wrist_shape_motion_v2"
                },
                output / "sign_model.pt"
            )
        else:
            stale_epochs += 1

        if stale_epochs >= 20:
            print("Stopping after 20 epochs without improvement.")
            break

    summary = {
        "seed": seed,
        "selected_epoch": best_epoch,
        "validation_accuracy": best_accuracy,
        "validation_loss": best_loss,
        "history": history
    }

    (output / "training_results.json").write_text(
        json.dumps(summary, indent=2)
    )

    print("\nV2 TRAINING COMPLETE")
    print("Selected epoch:", best_epoch)
    print(f"Validation accuracy: {best_accuracy:.1%}")
    print("Baseline validation accuracy: 70.2%")
    print("Saved to:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.seed)