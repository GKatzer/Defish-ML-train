# Final classifier: frozen DINOv2 + logistic regression (2026-09)

Replaces the deployed `fish_classifier.pt` (YOLOv8m-cls trained from scratch on 128 unique train crops) with frozen
DINOv2-base embeddings (`facebook/dinov2-base`, CLS + mean-patch token, hflip TTA) and a logistic regression head.
`many_fish` / `not_a_fish` are dropped as predicted classes — the server already discards them, and folding them into
the classifier only cost it two of nine classes' worth of training signal for no benefit downstream.

**Deployment note:** runs entirely on `onnxruntime` + `numpy` at inference, no torch or scikit-learn needed on the
target machine — see "ONNX export" below.

## Honest expected performance
Grouped stratified CV (4 folds x 8 seeds, split by source photo so a fish never has crops in both train and test
of a fold), 138 gold seven-class crops:

| | value |
|---|---|
| accuracy | 0.57 ± 0.04 |
| balanced accuracy | 0.58 |
| macro F1 | 0.56 |
| top-3 accuracy | 0.84 |

Deployed `fish_classifier.pt` gets ~0.36 on the equivalent unique-crop, non-leaked evaluation (see
`experiments/results/report.md`, part A, native 640 preprocessing — the closest match to the real server pipeline).

## Confidence gate
Instead of always forcing a label, predictions below a threshold are reported as `"uncertain": true`. The threshold
is picked from out-of-fold CV probabilities so that accuracy **among kept predictions** reaches a target; trade-off
(same CV run):

| threshold | kept | accuracy among kept |
|---|---|---|
| 0.00 (current server behaviour) | 100% | 0.57 |
| 0.45 | 92% | 0.60 |
| 0.65 | 70% | 0.67 |
| **0.83 (shipped default, target 0.75)** | **49%** | **0.75** |

`TARGET_PRECISION` in `train_final.py` controls this — lower it for more coverage at lower accuracy, raise it for
fewer but more trustworthy calls. 0.75 was picked as a reasonable "would you trust this in an app" bar; not
validated against a product requirement, worth revisiting with the user.

## Fixed: `plistophorosis` was confounded with species (neon tetra)
All 6 gold `plistophorosis` crops are neon tetras, and the 42 "healthy" gold crops contained zero neon tetra —
checked visually. The model had never seen a healthy neon tetra, so "is a neon tetra" and "has plistophorosis"
were perfectly correlated in training; its 98% CV recall was partly species recognition, not disease detection.

**Fix:** 5 crops of visibly healthy neon tetras (vivid colour, straight spine, no cysts — the visible signs of the
disease), cut out with our own detector from two Wikimedia Commons photos (CC BY-SA, credited in
`train_final.py`), added to the `healthy` class as a separate "silver" pool — always used in training, but not
part of the group-CV that estimates gold-only accuracy, and checked on its own via leave-one-out:

| | before (gold only) | after (+5 external tetra crops) |
|---|---|---|
| external healthy tetra photos misclassified as `plistophorosis` (leave-one-out) | 3 / 5 | 1 / 5 |

Meaningfully better, not fully solved. The one photo that still gets misclassified even leave-one-out has a
different visual style (macro close-up, grey studio-style background) matching the disease-photo domain more than
the other four (in-tank, green/white background) — likely a photographic-style confound layered on top of the
species one. More diverse healthy-tetra photos (in-tank, varied backgrounds) would likely close the rest of the gap;
not pursued further here. `dermatomycosis` and other visually-similar-looking diseases were not checked for the
same failure mode — worth a similar spot-check before trusting them fully.

## Tried and rejected: SAM-masked crops
See `SAM_EXPERIMENT_RESULT.md` — box-prompted SAM to flatten the background made accuracy worse (0.57 -> 0.45),
particularly for `fin_rot` (0.56 -> 0.35 recall), because it clips fin detail. Not adopted.

## ONNX export (no torch/scikit-learn needed to run it)
`export_onnx.py` exports the DINOv2 embedder to ONNX (pooling done inside the graph), verified against the torch
version (cosine similarity > 0.999 on real crops). `finalize_onnx_artifacts.py` extracts the trained
StandardScaler + LogisticRegression into plain numpy arrays (`head_numpy.npz`) — no scikit-learn needed either.
`infer_onnx.py` is the runtime: `onnxruntime` + `numpy` + `Pillow` only, smoke-tested with `torch`/`sklearn`
import-blocked to make sure nothing sneaks in.

Re-run both export scripts whenever `train_final.py` is re-run (it rewrites `meta.json`).

- **Size:** `embedder_dinov2_base.onnx` + `.onnx.data` together are **348 MB** (fp32). If that doesn't fit, fp16 or
  int8 quantization would roughly halve or quarter it — not done, ask if needed.
- **Speed (CPU):** ~130 ms per forward pass, ~260 ms per prediction (two passes: original + horizontal-flip TTA).
  That's slower than the ~200 ms detector pass; for an image with several fish this adds up. Dropping the hflip TTA
  halves it to ~130 ms/crop at an untested accuracy cost — an easy lever if latency becomes a problem.

## Files
- `train_final.py` — trains and evaluates; run to regenerate `artifacts/scaler.pkl` / `classifier.pkl` / `meta.json`.
- `export_onnx.py`, `finalize_onnx_artifacts.py` — produce the onnxruntime-only artifacts (see above).
- `infer.py` — torch-based reference implementation (development use).
- `infer_onnx.py` — the deployment-ready version: `onnxruntime` + `numpy` + `Pillow` only.
- `external_healthy_tetra/`, `external_healthy_tetra_embeddings.npy` — the 5 Wikimedia counterexample crops.
- `sam_crop_experiment.py`, `SAM_EXPERIMENT_RESULT.md` — the rejected SAM-masking experiment.
- `artifacts/` — `scaler.pkl`/`classifier.pkl` (joblib, dev), `head_numpy.npz` (numpy, deployment),
  `embedder_dinov2_base.onnx`(`.data`), `meta.json` (classes, threshold, expected CV metrics, ONNX file info).

## Status and open decisions
- Integrated into the inference service (`Defish-inference`: `inference/classifier.py`, `api/endpoints.py`), which runs the
  ONNX embedder and `head_numpy.npz` on `onnxruntime` + NumPy only.
- The confidence threshold (0.83, tuned for ~0.75 accuracy among kept predictions) is a choice, not validated against a
  product requirement.
