"""Evaluate fish detectors on the leakage-free splits built by build_splits.py.

Backends
  ultra       ultralytics predict (.pt or .onnx), gray 114 letterbox, as during training
  prod-white  the production server pipeline (the inference service, DEFISH_INFERENCE_DIR): letterbox to 960 with WHITE
              padding + YOLO_ONNX_Inference + unletterbox
  prod-gray   same as prod-white but with the gray 114 padding used in training

Metrics (IoU 0.5, one class): AP50, precision/recall at conf 0.25 (the server threshold), false positives per
image, best F1, and recall by ground-truth size (size = sqrt(w*h) at the 960 px letterbox scale).

Run:
    python experiments/detector/det_eval.py --baselines
    python experiments/detector/det_eval.py --model PATH --tag NAME [--backend ultra] [--imgsz 960]
"""
import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DETECTOR_V3 as V3, INFERENCE_DIR, MODELS_DIR  # noqa: E402  (see experiments/paths.py)

PROD = str(INFERENCE_DIR)  # the inference service: its inference.detector and utils.img are the production code
ML = MODELS_DIR            # weights of the baselines below
HERE = Path(__file__).parent
CACHE = HERE / "cache"
RESULTS = HERE / "results"
CACHE.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
BUCKETS = [("tiny<24", 0, 24), ("small 24-64", 24, 64), ("medium 64-192", 64, 192), ("large>=192", 192, 1e9)]


# ------------------------------------------------------------------ data
def load_split(split: str) -> pd.DataFrame:
    m = pd.read_csv(V3 / "manifest.csv")
    return m[m.split == split].reset_index(drop=True)


def read_gt(new_path: str, w: int, h: int) -> np.ndarray:
    lp = Path(new_path).parent.parent / "labels" / (Path(new_path).stem + ".txt")
    rows = [list(map(float, l.split()[1:5])) for l in lp.read_text().splitlines() if l.strip()]
    out = []
    for cx, cy, bw, bh in rows:
        out.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(out, dtype=np.float64).reshape(-1, 4)


# ------------------------------------------------------------------ predictors
def nms(dets: np.ndarray, iou_thr: float = 0.5) -> np.ndarray:
    if len(dets) == 0:
        return dets
    dets = dets[np.argsort(-dets[:, 4])]
    keep = []
    idx = np.arange(len(dets))
    while len(idx):
        i = idx[0]
        keep.append(i)
        if len(idx) == 1:
            break
        idx = idx[1:][iou_one_to_many(dets[i, :4], dets[idx[1:], :4]) <= iou_thr]
    return dets[keep]


def with_tiles(predict, long_side: int = 1920, tile: int = 960, stride: int = 720, margin: int = 4):
    """Full-image pass plus overlapping tiles of the image rescaled to `long_side` (SAHI-style).
    Tile detections that touch an interior tile border are dropped (they are cut-off fish)."""
    def run(img):
        h, w = img.shape[:2]
        full = predict(img)
        s = min(1.0, long_side / max(h, w))
        big = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
        H, W = big.shape[:2]
        if max(H, W) <= tile * 1.1:
            return full
        xs = list(range(0, max(W - tile, 0) + 1, stride)); ys = list(range(0, max(H - tile, 0) + 1, stride))
        if xs[-1] + tile < W: xs.append(W - tile)
        if ys[-1] + tile < H: ys.append(H - tile)
        out = [full]
        for y0 in ys:
            for x0 in xs:
                d = predict(np.ascontiguousarray(big[y0:y0 + tile, x0:x0 + tile]))
                if not len(d):
                    continue
                th, tw = min(tile, H - y0), min(tile, W - x0)
                bad = ((d[:, 0] < margin) & (x0 > 0)) | ((d[:, 1] < margin) & (y0 > 0)) | \
                      ((d[:, 2] > tw - margin) & (x0 + tw < W)) | ((d[:, 3] > th - margin) & (y0 + th < H))
                d = d[~bad]
                d[:, [0, 2]] = (d[:, [0, 2]] + x0) / s
                d[:, [1, 3]] = (d[:, [1, 3]] + y0) / s
                out.append(d)
        return nms(np.vstack(out), 0.5)
    return run


def make_predictor(backend: str, model_path: str, imgsz: int = 960):
    if backend.endswith("+tiles"):
        return with_tiles(make_predictor(backend[:-len("+tiles")], model_path, imgsz))
    if backend == "ultra":
        from ultralytics import YOLO
        model = YOLO(model_path)

        def predict(img_bgr):
            r = model.predict(img_bgr, imgsz=imgsz, conf=0.01, iou=0.45, max_det=300, verbose=False)[0]
            b = r.boxes
            return np.hstack([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None]]) if len(b) else np.zeros((0, 5))
        return predict

    sys.path.insert(0, PROD)
    from inference.detector import YOLO_ONNX_Inference  # the production code itself
    from utils.img import letterbox_resize, unletterbox_bbox
    det = YOLO_ONNX_Inference(model_path, img_size=imgsz, conf_thres=0.01, iou_thres=0.45)
    color = (255, 255, 255) if backend == "prod-white" else (114, 114, 114)

    def predict(img_bgr):
        lb, meta = letterbox_resize(img_bgr, target_size=imgsz, color=color)
        dets = det.predict_image(lb)
        return np.array([unletterbox_bbox(d["bbox"], meta) + [d["confidence"]] for d in dets]).reshape(-1, 5)
    return predict


def run_predictions(tag: str, split: str, backend: str, model_path: str, imgsz: int) -> list:
    cache = CACHE / f"pred_{tag}_{split}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    df = load_split(split)
    predict = make_predictor(backend, model_path, imgsz)
    out = []
    for i, r in enumerate(df.itertuples()):
        img = cv2.imread(r.new_path)
        h, w = img.shape[:2]
        out.append(dict(path=r.new_path, source=r.source, w=w, h=h, gt=read_gt(r.new_path, w, h), det=predict(img)))
        if (i + 1) % 40 == 0:
            print(f"  [{tag}/{split}] {i + 1}/{len(df)}", flush=True)
    cache.write_bytes(pickle.dumps(out))
    return out


# ------------------------------------------------------------------ metrics
def iou_one_to_many(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (a + b - inter + 1e-9)


def score(items: list, thr: float = 0.25) -> dict:
    """Greedy confidence-ordered matching at IoU 0.5; prefix property lets one pass serve every threshold."""
    dets = []  # (conf, image index, det index)
    for k, it in enumerate(items):
        for j, d in enumerate(it["det"]):
            dets.append((d[4], k, j))
    dets.sort(key=lambda t: -t[0])
    used = [np.zeros(len(it["gt"]), bool) for it in items]
    matched_conf = [np.full(len(it["gt"]), -1.0) for it in items]
    flags, confs = [], []
    for conf, k, j in dets:
        gt = items[k]["gt"]
        tp = False
        if len(gt):
            ious = iou_one_to_many(items[k]["det"][j][:4], gt)
            ious[used[k]] = -1
            g = int(ious.argmax())
            if ious[g] >= 0.5:
                used[k][g] = True
                matched_conf[k][g] = conf
                tp = True
        flags.append(tp)
        confs.append(conf)
    flags, confs = np.array(flags, bool), np.array(confs)
    n_gt = int(sum(len(it["gt"]) for it in items))
    tpc, fpc = np.cumsum(flags), np.cumsum(~flags)
    rec = tpc / max(n_gt, 1)
    prec = tpc / np.maximum(tpc + fpc, 1)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    n_at = int((confs >= thr).sum())
    tp_at = int(flags[:n_at].sum())
    res = dict(n_img=len(items), n_gt=n_gt, AP50=ap, P=tp_at / max(n_at, 1), R=tp_at / max(n_gt, 1),
               FP_per_img=(n_at - tp_at) / max(len(items), 1),
               bestF1=float(f1.max()) if len(f1) else 0.0, bestF1_thr=float(confs[int(f1.argmax())]) if len(f1) else 0.0)
    sizes, found = [], []
    for k, it in enumerate(items):
        s = np.sqrt((it["gt"][:, 2] - it["gt"][:, 0]) * (it["gt"][:, 3] - it["gt"][:, 1])) * 960 / max(it["w"], it["h"])
        sizes.append(s); found.append(matched_conf[k] >= thr)
    sizes, found = np.concatenate(sizes) if sizes else np.array([]), np.concatenate(found) if found else np.array([])
    for name, lo, hi in BUCKETS:
        m = (sizes >= lo) & (sizes < hi)
        res[f"R {name}"] = float(found[m].mean()) if m.any() else float("nan")
        res[f"n {name}"] = int(m.sum())
    return res


def evaluate(tag, split, backend, model_path, imgsz=960) -> list:
    items = run_predictions(tag, split, backend, model_path, imgsz)
    rows = [dict(model=tag, split=split, source="ALL", **score(items))]
    for src in sorted({it["source"] for it in items}):
        rows.append(dict(model=tag, split=split, source=src, **score([it for it in items if it["source"] == src])))
    return rows


def save(rows: list):
    f = RESULTS / "det_results.csv"
    new = pd.DataFrame(rows)
    if f.exists():
        old = pd.read_csv(f)
        old = old[~old.set_index(["model", "split"]).index.isin(new.set_index(["model", "split"]).index)]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(f, index=False)


def show(split: str = "test"):
    f = RESULTS / "det_results.csv"
    d = pd.read_csv(f)
    d = d[d.split == split]
    cols = ["model", "source", "n_img", "n_gt", "AP50", "P", "R", "FP_per_img", "bestF1"] + [f"R {b[0]}" for b in BUCKETS]
    print(f"\n=== {split} (P/R/FP at conf 0.25, IoU 0.5) ===")
    print(d[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))


BASELINES = [  # tag, backend, path
    ("PROD best.onnx (server pipeline, white pad)", "prod-white", str(INFERENCE_DIR / "models" / "best.onnx")),
    ("best.onnx (gray pad)", "prod-gray", str(INFERENCE_DIR / "models" / "best.onnx")),
    ("detector.onnx (Sep 7 2025)", "prod-gray", str(ML / "detector.onnx")),
    ("new_colab.onnx (Sep 9 2025)", "prod-gray", str(ML / "new_colab.onnx")),
    ("nout.pt (Sep 9 2025)", "ultra", str(ML / "nout.pt")),
    ("yolo.pt (May 2025, old server)", "ultra", str(ML / "yolo.pt")),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--tag")
    ap.add_argument("--backend", default="ultra")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--splits", nargs="+", default=["valid", "test"])
    a = ap.parse_args()
    jobs = BASELINES if a.baselines else [(a.tag, a.backend, a.model)]
    for tag, backend, path in jobs:
        for split in a.splits:
            print(f"evaluating {tag} on {split} ...", flush=True)
            save(evaluate(tag, split, backend, path, a.imgsz))
    for split in a.splits:
        show(split)


if __name__ == "__main__":
    main()
