from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "datasets" / "vehicles" / "data.yaml"


def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training on: {'GPU' if device == 0 else 'CPU'}")

    # Start from pretrained COCO weights (downloaded automatically on first run)
    model = YOLO("yolo11n.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=640,
        batch=8,
        patience=30,          # stop early if no improvement for 30 epochs
        device=device,
        project=str(ROOT / "runs" / "detect"),
        name="vehicles_v1",
        exist_ok=True,        # overwrite folder instead of creating vehicles_v12, v13...
        seed=42,
    )

    # Evaluate the best weights on the held-out test split
    best = YOLO(ROOT / "runs" / "detect" / "vehicles_v1" / "weights" / "best.pt")
    metrics = best.val(data=str(DATA_YAML), split="test")
    print(f"mAP50: {metrics.box.map50:.3f}  mAP50-95: {metrics.box.map:.3f}")


if __name__ == "__main__":   # required on Windows (dataloader uses multiprocessing)
    main()
