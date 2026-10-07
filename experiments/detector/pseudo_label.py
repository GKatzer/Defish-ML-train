"""Pseudo-label the unlabeled Reddit aquarium pool with a two-model agreement rule (self-training data).

Teachers: A = the new detector (best.pt), B = an independently trained one (nout.pt, close to the deployed model).
A box is accepted when A sees it with conf >= A_CONF and B sees an overlapping box (IoU >= 0.5) with conf >= B_CONF;
the accepted box is the confidence-weighted average. An image is dropped when either teacher has a strong detection
(conf >= CONFLICT_CONF) that the other does not confirm, because that may be a real but unconfirmed fish, and
training on it as background would teach the student to ignore fish. Images with no accepted box are not used.

Leakage control against the evaluation splits: pool images sharing a Reddit post id with any valid/test image, or
looking like one (dHash distance <= 6), are excluded; near-duplicates of labelled train images are excluded too.

    python experiments/detector/pseudo_label.py --teacher-a PATH
"""
import argparse
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DETECTOR_V3, DETECTOR_V3_PSEUDO, MODELS_DIR, REDDIT_ROOT  # noqa: E402  (see experiments/paths.py)

ROOT = REDDIT_ROOT
POOLS = [ROOT / "aquarium_new" / "gallery-dl", ROOT / "aquarium_top_1", ROOT / "aquarium_top_2"]
V3 = DETECTOR_V3
OUT = DETECTOR_V3_PSEUDO
ML = MODELS_DIR
A_CONF, B_CONF, CONFLICT_CONF = 0.5, 0.4, 0.5
EXT = {".jpg", ".jpeg", ".png", ".webp"}


def dhash(p) -> int:
    im = Image.open(p)
    im.draft("L", (128, 128))  # JPEG: decode at reduced size, avoids loading huge photos in full
    a = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    return int("".join("1" if x else "0" for x in (a[:, 1:] > a[:, :-1]).flatten()), 2)


def load_small(p, max_side=1280):
    im = Image.open(p)
    im.draft("RGB", (max_side, max_side))
    im = im.convert("RGB")
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def predict_all(model, paths, imgsz=960, batch=16):
    out = {}
    for i in range(0, len(paths), batch):
        chunk = [str(p) for p in paths[i:i + batch]]
        for p, r in zip(chunk, model.predict(chunk, imgsz=imgsz, conf=0.25, iou=0.45, verbose=False)):
            b = r.boxes
            out[p] = np.hstack([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None]]) if len(b) else np.zeros((0, 5))
        if (i // batch) % 20 == 0:
            print(f"  predicted {min(i + batch, len(paths))}/{len(paths)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-a", required=True)
    ap.add_argument("--teacher-b", default=str(ML / "nout.pt"))
    a = ap.parse_args()
    from ultralytics import YOLO

    m = pd.read_csv(V3 / "manifest.csv")
    eval_posts = set(m[m.split.isin(["valid", "test"]) & m.post.notna()].post)
    pool = [p for d in POOLS for p in d.rglob("*") if p.suffix.lower() in EXT]
    print("pool images:", len(pool))
    pool = [p for p in pool if re.split(r"[-_ ]", p.name)[0][:7] not in eval_posts]
    print("after removing eval Reddit posts:", len(pool))

    labelled = [(int(d), s) for d, s in zip(m.dh if "dh" in m else [], m.split)] if "dh" in m else None
    if labelled is None:  # manifest has no hashes: recompute for labelled images
        labelled = [(dhash(p), s) for p, s in zip(m.new_path, m.split)]
    eval_h = [h for h, s in labelled if s in ("valid", "test")]
    lab_h = [h for h, _ in labelled]
    keep = []
    for p in pool:
        try:
            h = dhash(p)
        except Exception:
            continue
        if any((h ^ e).bit_count() <= 6 for e in lab_h):  # duplicate of anything already labelled
            continue
        keep.append(p)
    print("after removing near-duplicates of labelled images:", len(keep))

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "images").mkdir(parents=True)
    (OUT / "labels").mkdir(parents=True)
    small_dir = OUT / "_pool_small"
    small_dir.mkdir()
    small = []
    for i, p in enumerate(keep):
        try:
            img = load_small(p)
        except Exception as e:  # unreadable / gigantic files are skipped
            print("  skip", p.name[:60], type(e).__name__)
            continue
        q = small_dir / f"{i:05d}.jpg"
        cv2.imwrite(str(q), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        small.append(q)
    print("downsized pool images:", len(small))
    keep = small
    A = predict_all(YOLO(a.teacher_a), keep)
    B = predict_all(YOLO(a.teacher_b), keep)

    n_img = n_box = n_conflict = n_empty = 0
    sizes = []
    for i, p in enumerate(keep):
        da, db = A[str(p)], B[str(p)]
        accepted, conflict = [], False
        used_b = set()
        for x in da:
            if x[4] < A_CONF:
                continue
            best, bj = 0, -1
            for j, y in enumerate(db):
                if y[4] >= B_CONF and iou(x, y) >= 0.5 and iou(x, y) > best:
                    best, bj = iou(x, y), j
            if bj >= 0:
                y = db[bj]; used_b.add(bj)
                w = x[4] + y[4]
                accepted.append((x[:4] * x[4] + y[:4] * y[4]) / w)
            else:
                conflict = True
        conflict |= any(y[4] >= CONFLICT_CONF and j not in used_b for j, y in enumerate(db))
        if conflict:
            n_conflict += 1
            continue
        if not accepted:
            n_empty += 1
            continue
        h, w = cv2.imread(str(p)).shape[:2]
        name = f"pseudo__{p.stem}"
        shutil.copy2(p, OUT / "images" / f"{name}.jpg")
        lines = []
        for x1, y1, x2, y2 in accepted:
            lines.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            sizes.append(np.sqrt((x2 - x1) * (y2 - y1)) * 960 / max(h, w))
        (OUT / "labels" / f"{name}.txt").write_text("\n".join(lines) + "\n")
        n_img += 1
        n_box += len(accepted)
    print(f"pseudo-labelled images={n_img} boxes={n_box} | dropped for conflict={n_conflict}, no confirmed box={n_empty}")
    if sizes:
        print("box size at 960 px, percentiles 5/25/50/75/95:", np.round(np.percentile(sizes, [5, 25, 50, 75, 95])).astype(int))
    shutil.rmtree(small_dir)
    base = f"path: {V3.as_posix()}\ntrain:\n  - train/images\n  - {(OUT / 'images').as_posix()}\nval: valid/images\ntest: test/images\nnc: 1\nnames: ['fish']\n"
    (V3 / "data_all_plus_pseudo.yaml").write_text(base)


if __name__ == "__main__":
    main()
