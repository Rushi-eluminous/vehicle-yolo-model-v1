# Vehicle Detection – Fine-tuned YOLO11n

YOLO11n model fine-tuned on a custom Roboflow dataset to detect vehicles in images and videos.

## Setup

```bash
git clone <repo-url>
cd fine-tune-model-1
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

## Usage

```bash
python scripts/predict.py --source path/to/video.mp4
python scripts/predict.py --source path/to/image.jpg --conf 0.5
python scripts/predict.py --source 0          # webcam
```

Annotated results are saved in `runs/detect/predict/`.

## Model

|               |                                                                           |
| ------------- | ------------------------------------------------------------------------- |
| Base model    | YOLO11n (COCO pretrained)                                                 |
| Classes       | 1 – `Vehicle`                                                             |
| Training data | 50 labelled images → 105 after augmentation (train/valid/test = 105/10/5) |
| Image size    | 640                                                                       |
| Epochs        | 100                                                                       |

### Test results

| Precision | Recall | mAP50 | mAP50-95 |
| --------- | ------ | ----- | -------- |
| 0.961     | 0.968  | 0.981 | 0.690    |

> Evaluated on a small test set (5 images); real-world accuracy may vary with camera angle, lighting and night footage.

## Retrain

1. Download the dataset from [Roboflow Universe](https://universe.roboflow.com/rushikesh-vyawhare/first-project-andua/dataset/2) in **YOLOv11** format.
2. Extract it to `datasets/vehicles/` and set `path:` in `data.yaml` to that folder.
3. Run `python scripts/train.py`

## Project structure

```
├── models/vehicles_v1.pt   # fine-tuned weights
├── scripts/train.py        # training + test evaluation
├── scripts/predict.py      # inference on images/videos/webcam
├── requirements.txt
└── README.md
```

## License

Model built with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0). Dataset: MIT.
