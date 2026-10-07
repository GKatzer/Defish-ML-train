"""Pool the three hand-labelled fish-detection datasets and re-split them without leakage.

Sources (all single class `fish`, YOLO txt labels):
  ds_v2        Reddit aquarium photos, mostly small fish            (553 imgs)
  hand_label   close-ups of (often sick) fish, larger boxes         (149 imgs)
  hl_null_tiny close-ups + fish-free aquascapes as negatives        (367 imgs)

Leakage control: images are merged into one group when they share a Reddit post id (ds_v2 only),
an identical md5, or a near-identical picture (dHash Hamming distance <= 6). Whole groups go to one split.

Output: $DEFISH_DATA_DIR/detector/v3/{train,valid,test}/{images,labels}, manifest.csv, data yamls and per-source lists.
Run:  python experiments/detector/build_splits.py
"""
import hashlib
import random
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA_DIR  # noqa: E402  (see experiments/paths.py)

ROOT = DATA_DIR
OUT = ROOT / "detector" / "v3"
SOURCES = {
    "ds_v2": ROOT / "Reddit-fishes" / "aquarium_new" / "ds_v2",
    "hand_label": ROOT / "Dataset_hand_label",
    "hl_null_tiny": ROOT / "Dataset_hl_null_tiny-fish",
}
FRACTION = {"valid": 0.15, "test": 0.15}
SEED = 0
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def dhash(p: Path) -> int:
    a = np.asarray(Image.open(p).convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    return int("".join("1" if x else "0" for x in (a[:, 1:] > a[:, :-1]).flatten()), 2)


def collect() -> pd.DataFrame:
    rows = []
    for src, root in SOURCES.items():
        for split in ("train", "valid", "test"):
            d = root / split / "images"
            if not d.exists():
                continue
            for p in sorted(d.iterdir()):
                if p.suffix.lower() not in IMG_EXT:
                    continue
                lp = root / split / "labels" / (p.stem + ".txt")
                boxes = [l.split() for l in lp.read_text().splitlines() if l.strip()] if lp.exists() else []
                rows.append(dict(source=src, old_split=split, path=str(p), label_path=str(lp), n_boxes=len(boxes),
                                 md5=hashlib.md5(p.read_bytes()).hexdigest(), dh=dhash(p),
                                 post=re.split(r"[-_ ]", p.name)[0][:7] if src == "ds_v2" else None))
    return pd.DataFrame(rows)


def group_ids(df: pd.DataFrame) -> np.ndarray:
    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    for key in ("md5", "post"):
        first = {}
        for i, v in enumerate(df[key]):
            if v is None or v != v:
                continue
            union(i, first.setdefault(v, i))
    dh = df.dh.tolist()
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if bin(dh[i] ^ dh[j]).count("1") <= 6:
                union(i, j)
    return np.array([find(i) for i in range(len(df))])


def assign_splits(df: pd.DataFrame) -> pd.Series:
    rng = random.Random(SEED)
    split_of_group = {}
    for src in SOURCES:
        sub = df[df.source == src]
        groups = list(sub.group.unique())
        rng.shuffle(groups)
        n = len(sub)
        want = {"test": FRACTION["test"] * n, "valid": FRACTION["valid"] * n}
        have = {"test": 0, "valid": 0}
        for g in groups:
            if g in split_of_group:
                sp = split_of_group[g]
            else:
                sp = "test" if have["test"] < want["test"] else "valid" if have["valid"] < want["valid"] else "train"
                split_of_group[g] = sp
            size = int((sub.group == g).sum())
            if sp in have:
                have[sp] += size
    return df.group.map(split_of_group)


def write(df: pd.DataFrame):
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "valid", "test"):
        (OUT / split / "images").mkdir(parents=True)
        (OUT / split / "labels").mkdir(parents=True)
    new_paths = []
    for i, r in enumerate(df.itertuples()):
        name = f"{r.source}__{i:04d}_{r.md5[:8]}"  # original names exceed the Windows path limit; see manifest.csv
        ext = Path(r.path).suffix.lower()
        dst_img = OUT / r.split / "images" / f"{name}{ext}"
        shutil.copy2(r.path, dst_img)
        lab = Path(r.label_path)
        lines = [l.split() for l in lab.read_text().splitlines() if l.strip()] if lab.exists() else []
        (OUT / r.split / "labels" / f"{name}.txt").write_text("".join(f"0 {' '.join(l[1:5])}\n" for l in lines))
        new_paths.append(str(dst_img))
    df = df.assign(new_path=new_paths)
    df.drop(columns=["dh"]).to_csv(OUT / "manifest.csv", index=False)

    def listing(sel):
        return "\n".join(sel.new_path.str.replace("\\", "/")) + "\n"

    (OUT / "train_ds_v2_only.txt").write_text(listing(df[(df.split == "train") & (df.source == "ds_v2")]))
    for split in ("valid", "test"):
        for src in SOURCES:
            (OUT / f"{split}_{src}.txt").write_text(listing(df[(df.split == split) & (df.source == src)]))
    base = f"path: {OUT.as_posix()}\ntrain: train/images\nval: valid/images\ntest: test/images\nnc: 1\nnames: ['fish']\n"
    (OUT / "data_all.yaml").write_text(base)
    (OUT / "data_ds_v2_only.yaml").write_text(base.replace("train: train/images", "train: train_ds_v2_only.txt"))


def main():
    df = collect()
    df["group"] = group_ids(df)
    df["split"] = assign_splits(df)
    print(f"images={len(df)} groups={df.group.nunique()}")
    print(df.groupby(["source", "split"]).agg(images=("path", "size"), boxes=("n_boxes", "sum"),
                                              empty=("n_boxes", lambda s: int((s == 0).sum()))).to_string())
    mixed = df.groupby("group").split.nunique()
    assert (mixed == 1).all(), "a group ended up in several splits"
    write(df)
    print("written to", OUT)


if __name__ == "__main__":
    main()
