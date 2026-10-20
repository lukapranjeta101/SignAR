from pathlib import Path
import json
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_model import ROOT
from train_model_v2 import prepare_v2

SEED = 42


def load_split(split, labels):
    paths = sorted((ROOT / "features" / split).glob("*.npz"))
    if not paths:
        raise RuntimeError(f"No feature files found for {split}")

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


class LSTMModel(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        # hidden has shape (num_layers * 2, batch, hidden_size):
        # index -2/-1 are the final forward/backward states.
        combined = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.classifier(self.dropout(combined))


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
    print("Technique: Bidirectional LSTM")
    print("Features per frame:", train_x.shape[-1])
    print("Training videos:", len(train_y))
    print("Validation videos:", len(val_y))
    print("Classes:", labels, flush=True)

    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=16,
        shuffle=True,
        num_workers=0
    )

    model = LSTMModel(train_x.shape[-1], len(labels)).to(device)
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

    destination = ROOT / "trained_models/trained_model_lstm"
    destination.mkdir(exist_ok=True)

    torch.save(
        {
            "state_dict": best_state,
            "labels": labels,
            "input_size": train_x.shape[-1],
            "best_epoch": best_epoch,
            "preprocessing": "wrist_shape_motion_v2",
            "technique": "bidirectional_lstm"
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
