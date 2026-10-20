from pathlib import Path
import copy
import json
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
STEPS = 48
SEED = 42


def prepare_video(path):
    with np.load(path, allow_pickle=False) as data:
        hands = data["hands"].copy()
        hand_mask = data["hand_present"].copy()
        pose = data["pose"].copy()
        pose_mask = data["pose_present"].copy()
        label = str(data["gloss"].item())

    # Use the span containing detected hands, plus a small margin.
    # Preserve gaps inside the span rather than removing those frames.
    active = np.flatnonzero(hand_mask.any(axis=1))
    if len(active) == 0:
        raise ValueError(f"No detected hands: {path.name}")

    start = max(0, int(active[0]) - 5)
    end = min(len(hands), int(active[-1]) + 6)

    hands = hands[start:end]
    hand_mask = hand_mask[start:end]
    pose = pose[start:end]
    pose_mask = pose_mask[start:end]

    # Estimate a stable shoulder center and width for this clip.
    left = pose[:, 11, :2]
    right = pose[:, 12, :2]
    centers = (left + right) / 2
    widths = np.linalg.norm(left - right, axis=1)

    reliable = (
        pose_mask
        & (pose[:, 11, 3] > 0.5)
        & (pose[:, 12, 3] > 0.5)
        & (widths > 0.02)
    )

    if not reliable.any():
        raise ValueError(f"No reliable shoulder reference: {path.name}")

    center = np.median(centers[reliable], axis=0)
    scale = float(np.median(widths[reliable]))

    # Start with image-plane x/y coordinates.
    # Hand and body depth values have different reference origins,
    # so this baseline does not combine their z coordinates.
    hand_xy = (hands[:, :, :, :2] - center) / scale
    hand_xy[~hand_mask] = 0

    # Nose, eyes, ears, shoulders, elbows and wrists.
    body_indices = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16]
    body = pose[:, body_indices]
    body_visible = pose_mask[:, None] & (body[:, :, 3] > 0.5)
    body_xy = (body[:, :, :2] - center) / scale
    body_xy[~body_visible] = 0

    length = len(hands)
    features = np.concatenate(
        [
            hand_xy.reshape(length, -1),
            body_xy.reshape(length, -1),
            hand_mask.astype(np.float32),
            body_visible.astype(np.float32)
        ],
        axis=1
    )

    # Sample whole frames so masks remain true/false.
    indices = np.rint(np.linspace(0, length - 1, STEPS)).astype(int)
    features = features[indices].astype(np.float32)

    if not np.isfinite(features).all():
        raise ValueError(f"Invalid coordinates: {path.name}")

    return features, label


def load_split(split, labels):
    paths = sorted((ROOT / "features" / split).glob("*.npz"))
    if not paths:
        raise RuntimeError(f"No feature files found for {split}")

    examples = []
    targets = []

    for path in paths:
        features, label = prepare_video(path)
        examples.append(features)
        targets.append(labels.index(label))

    return (
        torch.from_numpy(np.stack(examples)),
        torch.tensor(targets, dtype=torch.long)
    )


class SignModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(6),
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(64 * 6, num_classes)
        )

    def forward(self, x):
        return self.layers(x.transpose(1, 2))


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

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

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("Device:", device)
    print("Training videos:", len(train_y))
    print("Validation videos:", len(val_y))
    print("Classes:", labels)
    print("Test videos are not used in this script.", flush=True)

    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=16,
        shuffle=True,
        num_workers=0
    )

    model = SignModel(train_x.shape[-1], len(labels)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=0.001
    )
    criterion = nn.CrossEntropyLoss()

    val_x = val_x.to(device)
    val_y = val_y.to(device)

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    best_accuracy = 0.0
    stale_epochs = 0

    for epoch in range(1, 151):
        model.train()
        correct = 0
        seen = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
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
                (logits.argmax(dim=1) == val_y).float().mean().item()
            )

        print(
            f"Epoch {epoch:3d} | "
            f"train accuracy {correct / seen:.1%} | "
            f"val accuracy {val_accuracy:.1%} | "
            f"val loss {val_loss:.4f}",
            flush=True
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_accuracy = val_accuracy
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= 20:
            print("Stopping: validation loss has not improved for 20 epochs.")
            break

    destination = ROOT / "trained_models/trained_model"
    destination.mkdir(exist_ok=True)

    torch.save(
        {
            "state_dict": best_state,
            "labels": labels,
            "input_size": train_x.shape[-1],
            "steps": STEPS,
            "best_epoch": best_epoch,
            "preprocessing": "shoulder_xy_masks_v1"
        },
        destination / "sign_model.pt"
    )

    (destination / "labels.json").write_text(
        json.dumps(labels, indent=2)
    )

    print("\nTRAINING COMPLETE")
    print("Selected epoch:", best_epoch)
    print(f"Validation accuracy at that epoch: {best_accuracy:.1%}")
    print("Saved model:", destination / "sign_model.pt")


if __name__ == "__main__":
    main()