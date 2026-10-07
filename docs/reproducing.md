# Reproducing the work

What the scripts need, in which order they run, what each one writes, and what was and was not checked for this documentation. **None of the data are in the repository** ([data.md](data.md)); Data will be published: <!-- TODO: insert the link to the published data --> Until then the evaluations and the training cannot be re-run from a clone alone; what can be run without data is listed under [What can be run without data](#what-can-be-run-without-data).

- [Setup](#setup)
- [Paths](#paths)
- [Order of the scripts](#order-of-the-scripts)
- [Command-line reference](#command-line-reference)
- [What was checked](#what-was-checked)
- [What can be run without data](#what-can-be-run-without-data)

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt     # unpinned; add a CUDA build of torch for training
```

The experiments were run with a CUDA GPU (`device=0` in the detector training); `torch` from the default index is enough for everything else. Versions used for the checks of this documentation (CPU only, 2026-10-05): Python 3.12.3, torch 2.14.1, transformers 5.18.0, ultralytics 8.4.173, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.2, onnxruntime 1.30.0, open_clip_torch 3.3.0. The experiments themselves were run earlier, with versions that were not recorded (the exported detector's metadata names Ultralytics 8.4.157).

## Paths

The scripts read their locations from environment variables ([`experiments/paths.py`](../experiments/paths.py); `python experiments/paths.py` prints what they resolved to and whether each exists):

| Variable | Meaning | Default |
|---|---|---|
| `DEFISH_DATA_DIR` | root of the datasets, layout in [data.md](data.md#folder-layout-the-scripts-expect) | `./data` |
| `DEFISH_RUNS_DIR` | training runs of `det_train.py` | `./runs` |
| `DEFISH_MODELS_DIR` | weights of the baselines of `det_eval.py` (`detector.onnx`, `new_colab.onnx`, `nout.pt`, `yolo.pt`); `det_train.py` runs from here | `./models` |
| `DEFISH_INFERENCE_DIR` | a checkout of the inference service; `det_eval.py` imports its `inference.detector` and `utils.img`, and uses its `models/best.onnx` as the "PROD" baseline | `../Defish-inference` |
| `DEFISH_DEPLOYED_CLASSIFIER` | the old YOLO classifier `fish_classifier.pt`, baseline of `clf_frozen_vs_yolo.py` | `./models/fish_classifier.pt` |
| `HF_HOME` | Hugging Face cache (not set by the scripts) | the library default |

## Order of the scripts

Everything is run from the repository root. Outputs that are not committed (`.gitignore` excludes `*.csv`, `*.npy`, `*.pkl`, `*.onnx`, `*.pt` and others) are listed in the last column.

| Step | Command | Needs | Writes |
|---|---|---|---|
| 1. detector splits | `python experiments/detector/build_splits.py` | the three detector sets | `$DEFISH_DATA_DIR/detector/v3/{train,valid,test}`, `manifest.csv`, `data_*.yaml` |
| 2. detector training | `python experiments/detector/det_train.py --name v8s_pre_all --weights yolov8s.pt` | step 1, a GPU | `$DEFISH_RUNS_DIR/fish_det/<name>` |
| 3. detector evaluation | `python experiments/detector/det_eval.py --model PATH --tag NAME --backend prod-white` (or `--baselines`) | step 1, the inference service checkout | `experiments/detector/cache/`, `experiments/detector/results/det_results.csv` |
| 4. pseudo-labels (optional) | `python experiments/detector/pseudo_label.py --teacher-a PATH` | step 1, the Reddit pools, two detectors | `$DEFISH_DATA_DIR/detector/v3_pseudo`, `data_all_plus_pseudo.yaml` |
| 5. classifier comparison | `python experiments/clf_frozen_vs_yolo.py` | `Dataset_autocropped`, the deployed YOLO classifier, the external set, a download of the embedders | `experiments/results/report.md`, `experiments/manifest_unique_crops.csv`, embedding caches |
| 6. final classifier | `python experiments/classifier/train_final.py` | step 5's manifest and cache, the five external healthy-tetra crops in `experiments/classifier/external_healthy_tetra/` | `experiments/classifier/artifacts/{scaler.pkl,classifier.pkl,meta.json}` |
| 7. ONNX export | `python experiments/classifier/export_onnx.py` then `python experiments/classifier/finalize_onnx_artifacts.py` | step 6 | `artifacts/embedder_dinov2_base.onnx` (+ `.data`), `head_numpy.npz`, the ONNX fields of `meta.json` |
| 8. runtime | `python experiments/classifier/infer_onnx.py crop1.jpg crop2.jpg` | step 7 only: `onnxruntime`, NumPy, Pillow | the answer on the screen |
| 9. SAM experiment (rejected idea) | `python experiments/classifier/sam_crop_experiment.py` | step 5 and 6 | `sam_masked_crops/`, `sam_manifest.csv` |
| 10. probe of the deployed classifier | `python experiments/probe_deployed_on_external.py` | step 5's module | `experiments/results/probe_deployed_on_external.csv` |
| figures | `python docs/figures/make_figures.py` | matplotlib | `docs/media/*.png` |

Re-run steps 6 and 7 together whenever either changes: `train_final.py` rewrites `meta.json`, and `finalize_onnx_artifacts.py` merges the ONNX fields back in. After a re-run, copy `embedder_dinov2_base.onnx(.data)`, `head_numpy.npz` and `meta.json` into the inference service's `models/disease_classifier/`.

## Command-line reference

| Script | Options |
|---|---|
| `det_eval.py` | `--baselines` (six earlier models); or `--model PATH --tag NAME [--backend ultra\|prod-white\|prod-gray] [--imgsz 960]`; `--splits valid test` |
| `det_train.py` | `--name NAME` (required), `--weights yolov8s.pt` (a `*.pt` starts pretrained, a `*.yaml` from scratch), `--data PATH`, `--imgsz 960`, `--epochs 80`, `--batch 8`, `--patience 20` |
| `pseudo_label.py` | `--teacher-a PATH` (required), `--teacher-b PATH` (default `$DEFISH_MODELS_DIR/nout.pt`); thresholds are constants in the script (`A_CONF` 0.5, `B_CONF` 0.4, `CONFLICT_CONF` 0.5) |
| `build_splits.py`, `clf_frozen_vs_yolo.py`, `train_final.py`, `export_onnx.py`, `finalize_onnx_artifacts.py`, `sam_crop_experiment.py`, `probe_deployed_on_external.py` | no options; settings are constants at the top (`FRACTION`, `SEED`; `SEEDS`, `FOLDS`, `TARGET_PRECISION`, `EMBEDDER`, `AUX`) |
| `infer_onnx.py` | crop files as arguments; `FishDiseaseClassifierONNX(artifacts_dir).predict(image)` from Python (PIL image or BGR array) |

## What was checked

| Check | Result |
|---|---|
| every `experiments/` script and the archive scripts compile (`python -m py_compile`) | yes |
| `python experiments/paths.py` with defaults and with `DEFISH_DATA_DIR` and `DEFISH_INFERENCE_DIR` set | prints the resolved paths ([transcript](examples/transcripts/synthetic-smoke-test.txt) shows a run) |
| `--help` of `det_eval.py`, `det_train.py`, `pseudo_label.py` | prints the options |
| `build_splits.py` and `det_eval.py --backend prod-white` end to end on **synthetic** data (random noise with drawn rectangles) and a constant stand-in detector | runs; byte-identical images of different old splits land in one group and one split; the AP50 of the stand-in is 0.00, as it must be ([transcript](examples/transcripts/synthetic-smoke-test.txt)) |
| `infer_onnx.py` on two real crops, with the deployed ONNX files copied into `artifacts/` | runs; see [Quick start](../README.md#quick-start) |
| `docs/figures/make_figures.py` | writes the seven figures |
| `build_splits.py` and `det_eval.py` on the real data, any training script, `clf_frozen_vs_yolo.py`, `train_final.py`, the exports, the SAM experiment, `pseudo_label.py` | **not run**: they need data or weights that are not in the repository. After the move to environment variables they were compiled and, for those with a command line, asked for `--help`, nothing more |

## What can be run without data

```bash
export DEFISH_DATA_DIR=/tmp/defish-data DEFISH_INFERENCE_DIR=../Defish-inference
python docs/examples/synthetic_detector_data.py --standin-detector /tmp/standin_detector.onnx
python experiments/detector/build_splits.py
python experiments/detector/det_eval.py --model /tmp/standin_detector.onnx --tag standin --backend prod-white --splits valid test
```

This checks the plumbing (paths, grouping, the production-pipeline backend), not any model. `python docs/figures/make_figures.py` redraws the figures from the numbers typed into it, and `experiments/classifier/infer_onnx.py` runs the classifier on any crop once the ONNX files of a trained model are in `artifacts/`.
