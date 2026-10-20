import csv
import numpy as np
import torch

from train_model import ROOT
from train_model_v2 import prepare_v2
from train_lstm import LSTMModel


def main():
    checkpoint = torch.load(
        ROOT / "trained_models/trained_model_lstm/sign_model.pt",
        map_location="cpu",
        weights_only=True
    )

    labels = checkpoint["labels"]

    model = LSTMModel(checkpoint["input_size"], len(labels))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    paths = sorted((ROOT / "features/test").glob("*.npz"))
    if len(paths) != 190:
        raise RuntimeError(f"Expected 190 test files, found {len(paths)}")

    confusion = np.zeros((len(labels), len(labels)), dtype=int)
    results = []

    with torch.no_grad():
        for path in paths:
            features, actual = prepare_v2(path)
            inputs = torch.from_numpy(features).unsqueeze(0)
            logits = model(inputs)

            predicted_index = int(logits.argmax(dim=1).item())
            actual_index = labels.index(actual)
            predicted = labels[predicted_index]

            confusion[actual_index, predicted_index] += 1

            results.append({
                "file": path.name,
                "actual": actual,
                "predicted": predicted,
                "correct": actual == predicted
            })

    correct = int(np.trace(confusion))
    total = len(results)

    print("\nTEST RESULTS (LSTM)")
    print(f"Correct: {correct}/{total}")
    print(f"Test accuracy: {correct / total:.1%}")

    print("\nPER-SIGN RECALL")
    for index, label in enumerate(labels):
        count = int(confusion[index].sum())
        hits = int(confusion[index, index])
        print(f"{label:12} {hits:2}/{count:2} = {hits / count:.1%}")

    output = ROOT / "trained_models/trained_model_lstm"

    with (output / "test_predictions.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    with (output / "test_confusion_matrix.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["actual / predicted"] + labels)
        for label, counts in zip(labels, confusion):
            writer.writerow([label] + counts.tolist())

    print("\nEvaluation saved in:", output)


if __name__ == "__main__":
    main()
