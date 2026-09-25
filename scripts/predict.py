import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Detect vehicles in images/videos")
    parser.add_argument("--source", required=True, help="image, folder, video, or 0 for webcam")
    parser.add_argument("--model", default=str(ROOT / "models" / "vehicles_v1.pt"))
    parser.add_argument("--conf", type=float, default=0.4)
    args = parser.parse_args()

    model = YOLO(args.model)
    model.predict(source=args.source, conf=args.conf, save=True)


if __name__ == "__main__":
    main()
