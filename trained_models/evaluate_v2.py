import numpy as np
import torch

from train_model import ROOT, SignModel
from train_model_v2 import prepare_v2


def main():
    paths = sorted((ROOT / "features/test").glob("*.npz"))
    if len(paths) != 190:
        raise RuntimeError(f"Expected 190 test clips, found {len(paths)}")

    # Prepare each clip once using V2's features.
    prepared = [prepare_v2(path) for path in paths]
    inputs = torch.from_numpy(
        np.stack([features for features, label in prepared])
    )
    actual_labels = [label for features, label in prepared]

    accuracies = []

    for seed in [42, 7, 123]:
        checkpoint = torch.load(
            ROOT / f"trained_models/trained_model_v2/seed_{seed}/sign_model.pt",
            map_location="cpu",
            weights_only=True
        )

        if checkpoint["preprocessing"] != "wrist_shape_motion_v2":
            raise RuntimeError("Unexpected model preprocessing")

        labels = checkpoint["labels"]
        targets = torch.tensor(
            [labels.index(label) for label in actual_labels]
        )

        model = SignModel(checkpoint["input_size"], len(labels))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        with torch.no_grad():
            predictions = model(inputs).argmax(dim=1)

        correct = int((predictions == targets).sum().item())
        accuracy = correct / len(targets)
        accuracies.append(accuracy)

        print(f"\nSEED {seed}")
        print(f"Test accuracy: {accuracy:.1%} ({correct}/190)")

        for index, label in enumerate(labels):
            mask = targets == index
            total = int(mask.sum().item())
            hits = int((predictions[mask] == index).sum().item())
            print(f"{label:12} {hits}/{total} = {hits / total:.1%}")

    print(f"\nMean V2 test accuracy: {np.mean(accuracies):.1%}")
    print("Original baseline test accuracy: 68.4%")


if __name__ == "__main__":
    main()