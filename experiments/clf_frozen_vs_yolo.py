"""Frozen embeddings + logistic regression vs the deployed YOLO-cls classifier.

Rebuilds the classifier dataset without the `dup_` copies and without the valid/test overlap,
groups crops by their source photo, and compares:

  A. matched split: train on the 128 unique train crops, test on the 58 unique valid/test crops
       - deployed fish_classifier.pt with the server preprocessing (resize 224x224, imgsz=224)
       - deployed fish_classifier.pt with its native preprocessing (imgsz=640)
       - frozen embedding + logistic regression (DINOv2, BioCLIP)
  B. stratified group k-fold CV over all 189 unique crops (embedding models only)

Run with the shared venv:
    python experiments/clf_frozen_vs_yolo.py
"""
import hashlib
import json
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, normalize

warnings.filterwarnings("ignore")

from paths import CLASSIFIER_CROPS, DEPLOYED_CLASSIFIER, EXTERNAL_PANDA992  # noqa: E402  (see experiments/paths.py)

DATA = CLASSIFIER_CROPS
DEPLOYED = DEPLOYED_CLASSIFIER
HERE = Path(__file__).parent
OUT = HERE / "results"
CACHE = HERE / "cache"
OUT.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)

AUX = ["many_fish", "not_a_fish"]
PAD_RGB = (124, 116, 104)  # ImageNet mean colour
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ dataset manifest
def build_manifest() -> pd.DataFrame:
    rows = []
    for split in ["train", "valid", "test"]:
        for p in sorted((DATA / split).rglob("*.jpg")):
            name = re.sub(r"^dup_\d+_", "", p.name)
            m = re.match(r"(.*)_crop\d+_jpg\.rf\.[0-9a-f]+\.jpg$", name)
            rows.append(dict(
                path=str(p), label=p.parent.name, split=split, is_dup=p.name.startswith("dup_"),
                group=m.group(1) if m else name,
                md5=hashlib.md5(p.read_bytes()).hexdigest(),
            ))
    df = pd.DataFrame(rows)
    conflict = df.groupby("md5").label.nunique()
    assert (conflict == 1).all(), "same image stored under different class folders"
    uniq = df.sort_values(["is_dup", "split"]).groupby("md5", as_index=False).first()
    splits = df.groupby("md5").split.agg(lambda s: "+".join(sorted(set(s))))
    uniq["splits"] = uniq.md5.map(splits)
    uniq["origin"] = np.where(uniq.splits == "train", "train", "eval")  # valid/test are the same images
    print(f"files={len(df)}  unique={len(uniq)}  origin: {uniq.origin.value_counts().to_dict()}")
    return uniq.drop(columns=["split", "is_dup"]).reset_index(drop=True)


# ------------------------------------------------------------------ embedders
def pad_square(im: Image.Image) -> Image.Image:
    w, h = im.size
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), PAD_RGB)
    canvas.paste(im, ((s - w) // 2, (s - h) // 2))
    return canvas


def load_images(paths):
    return [pad_square(Image.open(p).convert("RGB")) for p in paths]


@torch.no_grad()
def embed_dinov2(images, name: str, size: int) -> np.ndarray:
    from transformers import AutoModel
    model = AutoModel.from_pretrained(name).to(DEVICE).eval().half()
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
    out = []
    for im in images:
        x = torch.from_numpy(np.asarray(im.resize((size, size), Image.BICUBIC))).permute(2, 0, 1)[None].float() / 255
        x = ((x.to(DEVICE) - mean) / std).half()
        feats = []
        for xi in (x, x.flip(-1)):  # hflip TTA
            h = model(pixel_values=xi).last_hidden_state
            feats.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1).float())
        out.append(torch.stack(feats).mean(0)[0].cpu().numpy())
    return np.stack(out)


@torch.no_grad()
def embed_bioclip(images, hub: str = "hf-hub:imageomics/bioclip") -> np.ndarray:
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(hub)
    model = model.to(DEVICE).eval().half()
    out = []
    for im in images:
        x = preprocess(im)[None].to(DEVICE).half()
        f = (model.encode_image(x) + model.encode_image(x.flip(-1))) / 2
        out.append(f.float().cpu().numpy()[0])
    return np.stack(out)


EMBEDDERS = {
    "dinov2_base_224": lambda ims: embed_dinov2(ims, "facebook/dinov2-base", 224),
    "dinov2_base_448": lambda ims: embed_dinov2(ims, "facebook/dinov2-base", 448),
    "bioclip_224": lambda ims: embed_bioclip(ims),
}


def get_embeddings(df: pd.DataFrame, tag: str = "gold") -> dict:
    ims = None
    embs = {}
    for name, fn in EMBEDDERS.items():
        f = CACHE / f"{tag}_{name}.npy"
        if f.exists() and len(np.load(f)) == len(df):
            embs[name] = np.load(f)
            continue
        ims = ims or load_images(df.path)
        print(f"embedding with {name} ...", flush=True)
        try:
            embs[name] = fn(ims)
            np.save(f, embs[name])
        except Exception as e:  # keep going if one model cannot be downloaded
            print(f"  !! {name} failed: {type(e).__name__}: {e}")
    return embs


# ------------------------------------------------------------------ classifiers / metrics
def fit_predict_lr(Xtr, ytr, Xte, C=1.0):
    sc = StandardScaler().fit(normalize(Xtr))
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
    clf.fit(sc.transform(normalize(Xtr)), ytr)
    return clf.predict(sc.transform(normalize(Xte))), clf.predict_proba(sc.transform(normalize(Xte))), clf.classes_


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def summarize(y_true, y_pred, classes_sick_vs_healthy=True):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    k = int((y_true == y_pred).sum())
    lo, hi = wilson(k, n)
    res = dict(n=n, acc=k / n, acc_ci=f"[{lo:.2f}, {hi:.2f}]",
               bal_acc=balanced_accuracy_score(y_true, y_pred),
               macro_f1=f1_score(y_true, y_pred, average="macro"))
    if classes_sick_vs_healthy and "healthy" in set(y_true):
        t, p = y_true == "healthy", y_pred == "healthy"
        res["sick_recall"] = float(((~t) & (~p)).sum() / max((~t).sum(), 1))
        res["healthy_recall"] = float((t & p).sum() / max(t.sum(), 1))
    return res


# ------------------------------------------------------------------ deployed YOLO-cls
def predict_yolo(paths, mode: str):
    from ultralytics import YOLO
    model = YOLO(str(DEPLOYED))
    names = model.names
    probs = []
    for p in paths:
        if mode == "server":  # exactly what accounts/services/yolo.py does
            im = Image.open(p).convert("RGB").resize((224, 224))
            r = model(im, imgsz=224, verbose=False)[0]
        else:  # native: what the model saw in training (imgsz 640)
            r = model(p, imgsz=640, verbose=False)[0]
        probs.append(r.probs.data.cpu().numpy())
    return np.stack(probs), [names[i] for i in range(len(names))]


def restrict(probs, names, keep):
    idx = [names.index(c) for c in keep]
    return np.array(keep)[probs[:, idx].argmax(1)]


# ------------------------------------------------------------------ experiments
def experiment_a(df, embs, classes7):
    tr, te = df[df.origin == "train"], df[df.origin == "eval"]
    leak = te.group.isin(set(tr.group))
    subsets = {"all eval": te, "eval, photo not in train": te[~leak]}
    print(f"\n[A] train={len(tr)} eval={len(te)} (eval crops whose source photo is also in train: {int(leak.sum())})")
    rows = []
    yolo = {m: predict_yolo(te.path, m) for m in ("server", "native")}
    for sname, sub in subsets.items():
        pos = np.where(te.index.isin(sub.index))[0]
        y9 = sub.label.values
        m7 = ~np.isin(y9, AUX)
        for m, (pr, names) in yolo.items():
            p9 = np.array(names)[pr[pos].argmax(1)]
            p7 = restrict(pr[pos][m7], names, classes7)
            rows.append(dict(system=f"YOLO-cls deployed ({m} preproc)", subset=sname, task="9-class", **summarize(y9, p9)))
            rows.append(dict(system=f"YOLO-cls deployed ({m} preproc)", subset=sname, task="7-class", **summarize(y9[m7], p7)))
        for en, X in embs.items():
            Xtr, Xte = X[tr.index], X[sub.index]
            p9, _, _ = fit_predict_lr(Xtr, tr.label.values, Xte)
            rows.append(dict(system=f"{en} + LR", subset=sname, task="9-class", **summarize(y9, p9)))
            tr7 = tr[~tr.label.isin(AUX)]
            p7, _, _ = fit_predict_lr(X[tr7.index], tr7.label.values, Xte[m7])
            rows.append(dict(system=f"{en} + LR", subset=sname, task="7-class", **summarize(y9[m7], p7)))
    return pd.DataFrame(rows)


def experiment_b(df, embs, seeds=5, folds=4, C=1.0):
    rows, cms = [], {}
    for en, X in embs.items():
        for task in ("9-class", "7-class", "healthy-vs-sick"):
            sub = df if task == "9-class" else df[~df.label.isin(AUX)]
            y = sub.label.values if task != "healthy-vs-sick" else np.where(sub.label.values == "healthy", "healthy", "sick")
            per_seed = []
            for seed in range(seeds):
                oof = np.empty(len(sub), dtype=object)
                sgkf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
                for tri, tei in sgkf.split(sub.index, y, sub.group):
                    oof[tei], _, _ = fit_predict_lr(X[sub.index[tri]], y[tri], X[sub.index[tei]], C=C)
                per_seed.append(summarize(y, oof, classes_sick_vs_healthy=(task != "healthy-vs-sick")))
                if seed == 0:
                    cms[(en, task)] = pd.crosstab(pd.Series(y, name="true"), pd.Series(oof, name="pred"))
            agg = {k: float(np.mean([s[k] for s in per_seed])) for k in ("acc", "bal_acc", "macro_f1")}
            agg["acc_sd"] = float(np.std([s["acc"] for s in per_seed]))
            rows.append(dict(embedder=en, task=task, n=len(sub), **agg))
    return pd.DataFrame(rows), cms


EXT = EXTERNAL_PANDA992
# silver classes that have a plausible counterpart in our label set (parasitic/bacterial/viral have none)
SILVER_MAP = {"healthy_fish": "healthy", "fungal_diseases_saprolegniasis": "dermatomycosis"}


def load_external() -> pd.DataFrame:
    m = pd.read_csv(EXT / "manifest.csv")
    conflict = m.groupby("md5").label.nunique()
    m = m[m.md5.map(conflict) == 1].drop_duplicates("md5").reset_index(drop=True)
    m["path"] = m.path.map(lambda p: str(EXT / p))
    return m


def experiment_c(gold, gold_embs, ext, ext_embs, seeds=5, folds=4, n_silver=100):
    """C1: does a gold-trained model call outside 'healthy' fish healthy?  C2: does adding silver data help on gold?"""
    g = gold[~gold.label.isin(AUX)]
    probe_rows, cv_rows = [], []
    rng = np.random.default_rng(0)
    for en, X in gold_embs.items():
        Xe = ext_embs[en]
        # C1: train on all gold (7 classes), predict every silver image
        pred, _, _ = fit_predict_lr(X[g.index], g.label.values, Xe)
        for slab, sub in ext.groupby("label"):
            p = pred[sub.index]
            probe_rows.append(dict(embedder=en, silver_class=slab, n=len(sub),
                                   predicted_healthy=float((p == "healthy").mean()),
                                   top_pred=pd.Series(p).value_counts().index[0]))
        # C2: grouped CV on gold with silver added to the training side only
        configs = {"gold only": {}, f"+healthy x{n_silver}": {"healthy_fish": n_silver},
                   f"+healthy x{n_silver} +fungal->dermatomycosis x{n_silver}": {"healthy_fish": n_silver, "fungal_diseases_saprolegniasis": n_silver}}
        for cname, take in configs.items():
            idx = np.concatenate([rng.choice(ext.index[ext.label == l], n, replace=False) for l, n in take.items()]) if take else np.array([], int)
            ys = np.array([SILVER_MAP[ext.label[i]] for i in idx], dtype=object)
            res = []
            for seed in range(seeds):
                oof = np.empty(len(g), dtype=object)
                for tri, tei in StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed).split(g.index, g.label.values, g.group):
                    Xtr = np.vstack([X[g.index[tri]], Xe[idx]]) if len(idx) else X[g.index[tri]]
                    ytr = np.concatenate([g.label.values[tri], ys]) if len(idx) else g.label.values[tri]
                    oof[tei], _, _ = fit_predict_lr(Xtr, ytr, X[g.index[tei]])
                res.append(summarize(g.label.values, oof))
            cv_rows.append(dict(embedder=en, config=cname, **{k: float(np.mean([r[k] for r in res])) for k in ("acc", "bal_acc", "macro_f1", "sick_recall", "healthy_recall")}))
    return pd.DataFrame(probe_rows), pd.DataFrame(cv_rows)


def fmt(d: pd.DataFrame) -> str:
    d = d.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: f"{v:.2f}")
    head = "| " + " | ".join(d.columns) + " |\n|" + "---|" * len(d.columns) + "\n"
    return head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in d.values)


def main():
    manifest = build_manifest()
    manifest.to_csv(HERE / "manifest_unique_crops.csv", index=False)
    print(manifest.groupby(["label", "origin"]).size().unstack(fill_value=0))
    embs = get_embeddings(manifest)
    classes7 = sorted(c for c in manifest.label.unique() if c not in AUX)

    a = experiment_a(manifest, embs, classes7)
    b, cms = experiment_b(manifest, embs)
    ext = load_external()
    c1, c2 = experiment_c(manifest, embs, ext, get_embeddings(ext, "panda992"))

    md = ["# Frozen embeddings vs deployed YOLO-cls\n",
          f"Unique crops: {len(manifest)} (train {int((manifest.origin=='train').sum())}, eval {int((manifest.origin=='eval').sum())}).\n",
          "## A. Matched split (train on 128 unique train crops, test on unique valid/test crops)\n", fmt(a), "\n",
          "## B. Stratified group 4-fold CV x5 seeds over all unique crops (LR, C=1)\n", fmt(b), "\n",
          "## C1. Gold-trained model (7 classes) applied to external panda992 images\n"
          "`predicted_healthy` = share of images the model calls healthy. For `healthy_fish` it should be high.\n", fmt(c1), "\n",
          "## C2. Does adding silver data help? Group CV on gold, silver added to training folds only\n", fmt(c2), "\n"]
    for (en, task), cm in cms.items():
        if task != "healthy-vs-sick" and en == "dinov2_base_448":
            md += [f"### Confusion matrix (seed 0, out-of-fold): {en}, {task}\n", "```", cm.to_string(), "```\n"]
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
