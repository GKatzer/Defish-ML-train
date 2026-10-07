"""Train a fish detector on the leakage-free splits.

Same recipe as archive/2025-exploration/sssss.py (imgsz 960, mosaic, scale 0.5, hsv) except: starts from pretrained weights,
saves runs to $DEFISH_RUNS_DIR/fish_det, and drops copy_paste (it needs polygon labels, boxes-only data ignore it).

    python experiments/detector/det_train.py --name v8s_pre_all --weights yolov8s.pt
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DETECTOR_V3 as V3, MODELS_DIR, RUNS_DIR  # noqa: E402  (see experiments/paths.py)

ML = MODELS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--weights", default="yolov8s.pt")  # a *.pt starts pretrained, a *.yaml trains from scratch
    ap.add_argument("--data", default=str(V3 / "data_all.yaml"))
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patience", type=int, default=20)
    a = ap.parse_args()

    ML.mkdir(parents=True, exist_ok=True)
    os.chdir(ML)  # ultralytics' AMP check looks for yolo11n.pt in the cwd and would otherwise try to download it
    from ultralytics import YOLO

    weights = a.weights if Path(a.weights).is_absolute() or a.weights.endswith(".yaml") else str(ML / a.weights)
    YOLO(weights).train(
        data=a.data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, patience=a.patience,
        project=str(RUNS_DIR.resolve() / "fish_det"), name=a.name, exist_ok=False,
        mosaic=1.0, close_mosaic=10, scale=0.5, translate=0.1, degrees=0, fliplr=0.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        cache=False, workers=4, seed=0, amp=True, plots=True, device=0,
    )


if __name__ == "__main__":
    main()
