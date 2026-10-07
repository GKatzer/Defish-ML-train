"""Final disease classifier: frozen DINOv2 embeddings + logistic regression, replacing YOLO-cls-from-scratch.

Changes vs the deployed `fish_classifier.pt`:
  - drops `many_fish` / `not_a_fish` as predicted classes (7 classes: healthy + 6 diseases)
  - reports a calibrated "uncertain" gate instead of always forcing a top-1 label
  - exposes top-3, not just top-1

Evaluation is honest: expected accuracy comes from grouped (by source photo) stratified CV on the 138 seven-class
gold crops (same protocol as ML/experiments/clf_frozen_vs_yolo.py, part B). The deployed artifact is then re-fit on
ALL 138 crops (no holdout) because gold data is too scarce to sacrifice any of it — CV numbers are the honest
estimate of what that final fit will do on new photos.

Confidence gate: chosen from out-of-fold CV probabilities so that among predictions the model KEEPS (max prob above
the threshold), out-of-fold accuracy reaches TARGET_PRECISION. Predictions below the threshold are reported as
"uncertain" instead of a forced guess.

Outputs -> ML/experiments/classifier/artifacts/:
  scaler.pkl, classifier.pkl (StandardScaler + LogisticRegression, joblib)
  meta.json (embedder name/size, class list, chosen threshold, CV metrics)
  infer.py (standalone predict(image_bgr) -> {label, confidence, uncertain, top3}, mirrors the server's classifier.py shape)

Run:  python experiments/classifier/train_final.py
"""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, normalize
from transformers import AutoModel

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
EXP = HERE.parent
ART = HERE / "artifacts"
ART.mkdir(exist_ok=True)

EMBEDDER = "dinov2_base_224"  # tied with 448 on 7-class CV (.57 vs .58) and healthy-vs-sick (.81 both); smaller/faster
AUX = ["many_fish", "not_a_fish"]
DROP = AUX
TARGET_PRECISION = 0.75  # among kept (non-"uncertain") predictions
SEEDS, FOLDS = 8, 4

# plistophorosis: all 6 gold crops are neon tetras, and there was zero neon tetra among the 42 "healthy" gold crops
# (checked visually) -- the model had never seen a counterexample, so "is a neon tetra" and "has plistophorosis"
# were perfectly correlated in training. Fix: 5 crops of visibly healthy neon tetras (no colour loss, no cysts, no
# spine curvature -- the visible signs of the disease), cropped with our own detector from two CC-BY-SA Wikimedia
# Commons photos, added to the "healthy" class. Not fish-keeper verified like the gold set, so kept as a separate
# "silver" pool: always used in training, never held out by the grouped CV (that CV still reports the honest
# gold-only number), and checked on its own via leave-one-out below.
EXTERNAL_HEALTHY_DIR = HERE / "external_healthy_tetra"
EXTERNAL_HEALTHY_SOURCES = {
    "wikimedia_tetra_01.jpg": ("wikimedia_1", "Paracheirodon_innesi_2.jpg, Ernst Schuette 2004, CC BY-SA 3.0/GFDL"),
    "wikimedia_tetra_02.jpg": ("wikimedia_2", "Paracheirodon_innesi.jpg, Leo D'lion 2006, CC BY-SA 2.0"),
    "wikimedia_tetra_03.jpg": ("wikimedia_2", "Paracheirodon_innesi.jpg, Leo D'lion 2006, CC BY-SA 2.0"),
    "wikimedia_tetra_04.jpg": ("wikimedia_2", "Paracheirodon_innesi.jpg, Leo D'lion 2006, CC BY-SA 2.0"),
    "wikimedia_tetra_05.jpg": ("wikimedia_2", "Paracheirodon_innesi.jpg, Leo D'lion 2006, CC BY-SA 2.0"),
}


def embed_dinov2(paths, size=224, name="facebook/dinov2-base"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(name).to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def pad_square(im):
        w, h = im.size
        s = max(w, h)
        c = Image.new("RGB", (s, s), (124, 116, 104))
        c.paste(im, ((s - w) // 2, (s - h) // 2))
        return c

    out = []
    with torch.no_grad():
        for p in paths:
            im = pad_square(Image.open(p).convert("RGB")).resize((size, size), Image.BICUBIC)
            x = torch.from_numpy(np.array(im)).permute(2, 0, 1)[None].float().to(device) / 255
            x = (x - mean) / std
            feats = [torch.cat([h[:, 0], h[:, 1:].mean(1)], 1) for h in
                     (model(pixel_values=xi).last_hidden_state for xi in (x, x.flip(-1)))]
            out.append(torch.stack(feats).mean(0)[0].cpu().numpy())
    return np.stack(out)


def load_external_healthy():
    paths = list(EXTERNAL_HEALTHY_SOURCES)
    cache = HERE / "external_healthy_tetra_embeddings.npy"
    if cache.exists():
        X = np.load(cache)
    else:
        X = embed_dinov2([EXTERNAL_HEALTHY_DIR / p for p in paths])
        np.save(cache, X)
    groups = [EXTERNAL_HEALTHY_SOURCES[p][0] for p in paths]
    return paths, X, np.array(groups)


def fit_predict(Xtr, ytr, Xte, C=1.0):
    sc = StandardScaler().fit(normalize(Xtr))
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
    clf.fit(sc.transform(normalize(Xtr)), ytr)
    proba = clf.predict_proba(sc.transform(normalize(Xte)))
    return proba, clf.classes_


def cv_out_of_fold(X, y, groups):
    """Out-of-fold probabilities for every gold crop, pooled over several seeds for a stable threshold estimate."""
    rows = []
    for seed in range(SEEDS):
        for tri, tei in StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=seed).split(X, y, groups):
            proba, classes = fit_predict(X[tri], y[tri], X[tei])
            for k, i in enumerate(tei):
                rows.append(dict(seed=seed, idx=i, true=y[i], pred=classes[proba[k].argmax()],
                                 conf=proba[k].max(), proba=dict(zip(classes, proba[k]))))
    return pd.DataFrame(rows)


def pick_threshold(oof: pd.DataFrame, target_precision: float) -> float:
    """Smallest confidence threshold whose kept-prediction accuracy (averaged over seeds) reaches target_precision."""
    for thr in np.arange(0.30, 0.95, 0.01):
        kept = oof[oof.conf >= thr]
        if len(kept) < 20:
            continue
        acc = (kept.true == kept.pred).mean()
        if acc >= target_precision:
            return round(float(thr), 2)
    return 0.9  # fallback: nothing reached the target, gate hard


def leave_one_out_external(gold_X, gold_y, gold_groups, ext_paths, ext_X, ext_groups):
    """For each external healthy-tetra crop, train on gold + every OTHER external crop, predict this one.
    This is the direct test of the species-confound fix: does the model now call a healthy tetra 'healthy'
    (or at least not 'plistophorosis'), instead of blindly keying off the neon-tetra shape."""
    rows = []
    for i in range(len(ext_paths)):
        others = [j for j in range(len(ext_paths)) if j != i]
        Xtr = np.vstack([gold_X, ext_X[others]])
        ytr = np.concatenate([gold_y, np.array(["healthy"] * len(others))])
        proba, classes = fit_predict(Xtr, ytr, ext_X[i:i + 1])
        proba = proba[0]
        order = np.argsort(-proba)
        rows.append(dict(file=ext_paths[i], pred=classes[order[0]], pred_conf=float(proba[order[0]]),
                         plistophorosis_prob=float(proba[list(classes).index("plistophorosis")])))
    return pd.DataFrame(rows)


def main():
    manifest = pd.read_csv(EXP / "manifest_unique_crops.csv")
    X_all = np.load(EXP / "cache" / f"gold_{EMBEDDER}.npy")
    assert len(X_all) == len(manifest)

    keep_mask = ~manifest.label.isin(DROP)
    df = manifest[keep_mask].reset_index(drop=True)
    X = X_all[keep_mask.values]
    y = df.label.values
    groups = df.group.values
    classes = sorted(set(y))
    print(f"gold crops: {len(df)} | classes ({len(classes)}): {classes} | dropped: {DROP}")
    print(df.groupby("label").size().to_string())

    ext_paths, ext_X, ext_groups = load_external_healthy()
    print(f"\nexternal silver 'healthy' crops (Wikimedia neon tetra, not fish-keeper verified): {len(ext_paths)}")

    print("\n--- leave-one-out check on the external tetra crops (does the species confound persist?) ---")
    loo = leave_one_out_external(X, y, groups, ext_paths, ext_X, ext_groups)
    print(loo.to_string(index=False))
    if (loo.pred == "plistophorosis").any():
        print("WARNING: at least one healthy external tetra was still called plistophorosis -- confound not fixed, "
              "reconsider dropping the class instead.")
    else:
        print("OK: none of the external healthy tetra crops were called plistophorosis.")

    # ---- honest expected performance: grouped CV on GOLD ONLY (external silver data is not part of this estimate)
    oof = cv_out_of_fold(X, y, groups)
    per_seed = oof.groupby("seed").apply(lambda g: pd.Series(dict(
        acc=(g.true == g.pred).mean(), bal_acc=balanced_accuracy_score(g.true, g.pred),
        macro_f1=f1_score(g.true, g.pred, average="macro"),
        top3=np.mean([t in sorted(p, key=p.get, reverse=True)[:3] for t, p in zip(g.true, g.proba)]))))
    print("\nCV over", SEEDS, "seeds x", FOLDS, "folds (grouped by source photo):")
    print(per_seed.mean().round(3).to_string(), "\n(sd)\n", per_seed.std().round(3).to_string())

    thr = pick_threshold(oof, TARGET_PRECISION)
    kept = oof[oof.conf >= thr]
    print(f"\nconfidence gate: keep if top prob >= {thr}")
    print(f"  kept {len(kept)}/{len(oof)} ({len(kept) / len(oof):.0%}) of CV predictions, "
          f"accuracy among kept = {(kept.true == kept.pred).mean():.2f}")
    print(f"  dropped {len(oof) - len(kept)} as 'uncertain', accuracy among those (if forced) = "
          f"{(oof[oof.conf < thr].true == oof[oof.conf < thr].pred).mean():.2f}")
    cm = pd.crosstab(oof.true, oof.pred, normalize="index")
    print("\nconfusion (row-normalized, pooled over seeds):\n", (cm * 100).round(0).astype(int).to_string())

    # ---- final artifact: refit on ALL gold crops + external silver healthy-tetra crops (no holdout)
    Xfit = np.vstack([X, ext_X])
    yfit = np.concatenate([y, np.array(["healthy"] * len(ext_paths))])
    sc = StandardScaler().fit(normalize(Xfit))
    clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(sc.transform(normalize(Xfit)), yfit)
    joblib.dump(sc, ART / "scaler.pkl")
    joblib.dump(clf, ART / "classifier.pkl")
    meta = dict(embedder=EMBEDDER, classes=list(clf.classes_), confidence_threshold=thr,
               expected_cv_accuracy=round(float(per_seed.acc.mean()), 3),
               expected_cv_accuracy_sd=round(float(per_seed.acc.std()), 3),
               expected_cv_top3_accuracy=round(float(per_seed.top3.mean()), 3),
               n_train_crops=int(len(df)), n_external_silver_crops=int(len(ext_paths)),
               trained_on=f"all {len(classes)} gold classes + {len(ext_paths)} external silver 'healthy' crops, "
                          f"no holdout (dropped: {DROP})")
    (ART / "meta.json").write_text(json.dumps(meta, indent=2))
    print("\nsaved artifacts to", ART)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
