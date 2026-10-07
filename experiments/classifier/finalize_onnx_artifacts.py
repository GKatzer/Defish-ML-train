"""Glue step: extract the trained scaler+LR into plain numpy (head_numpy.npz) and merge the ONNX embedder info
back into meta.json (train_final.py rewrites meta.json and does not know about the ONNX export).

Run AFTER train_final.py and export_onnx.py, whenever either is re-run:
    python experiments/classifier/finalize_onnx_artifacts.py
"""
import json
from pathlib import Path

import joblib
import numpy as np

ART = Path(__file__).parent / "artifacts"


def main():
    sc = joblib.load(ART / "scaler.pkl")
    clf = joblib.load(ART / "classifier.pkl")
    np.savez(ART / "head_numpy.npz", scaler_mean=sc.mean_, scaler_scale=sc.scale_,
             lr_coef=clf.coef_, lr_intercept=clf.intercept_,
             classes=np.array(clf.classes_, dtype="<U32"))  # fixed-width unicode: loadable without allow_pickle

    onnx_path = ART / "embedder_dinov2_base.onnx"
    if not onnx_path.exists():
        print("no embedder_dinov2_base.onnx found -- run export_onnx.py first"); return
    meta = json.loads((ART / "meta.json").read_text())
    meta["onnx_embedder"] = onnx_path.name
    meta["onnx_embedder_size_mb"] = round((onnx_path.stat().st_size + (onnx_path.with_suffix(".onnx.data")).stat().st_size) / 1e6, 1) \
        if (ART / "embedder_dinov2_base.onnx.data").exists() else round(onnx_path.stat().st_size / 1e6, 1)
    (ART / "meta.json").write_text(json.dumps(meta, indent=2))
    print("wrote head_numpy.npz and updated meta.json:")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
