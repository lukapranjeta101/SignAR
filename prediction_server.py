from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "trained_models"))

from train_model import ROOT, SignModel, prepare_video


def main():
    checkpoint = torch.load(
        ROOT / "trained_models/trained_model/sign_model.pt",
        map_location="cpu",
        weights_only=True
    )

    labels = checkpoint["labels"]
    model = SignModel(checkpoint["input_size"], len(labels))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Use a validation clip for the demonstration.
    # Selection uses its actual label, not whether the prediction is correct.
    selected = None

    for path in sorted((ROOT / "features/val").glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            if str(data["gloss"].item()) == "HELLO":
                selected = path
                break

    if selected is None:
        raise RuntimeError("No HELLO validation clip found.")

    features, actual = prepare_video(selected)
    inputs = torch.from_numpy(features).unsqueeze(0)

    with torch.no_grad():
        scores = torch.softmax(model(inputs), dim=1)[0]
        index = int(scores.argmax().item())

    prediction = labels[index]
    score = float(scores[index].item())

    message = (
        "PRERECORDED CLIP DEMO\n"
        f"Predicted: {prediction}\n"
        f"Actual label: {actual}\n"
        f"Model score: {score:.1%}"
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            content = message.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)

    print("Clip:", selected.name)
    print(message)
    print("\nPrediction server ready on port 8000.")
    print("Press Play in Unity. Press Control + C here to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()