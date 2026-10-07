# 2025 exploration (archive)

Notebooks and scripts from the first phase of the project. They are kept for the record. They contain paths of the author's machine, they were run once, interactively, and **they are not the reproducible part of the repository**: the reproducible, evaluated work is in [`../../experiments/`](../../experiments/). The notebooks may still carry saved outputs (including photographs from third-party posts); strip them (`nbstripout`) before any further sharing.

| File | What it was |
|---|---|
| `autolabel.ipynb` | tries the first detector on a Reddit aquarium photo to auto-label (confidence 0.5, IoU 0.1) |
| `autolabel_for_spade.ipynb` | crops fish with the detector (confidence 0.7, at least 50 px, 256×256), embeds the crops and clusters them (K-means after PCA) to look at the groups |
| `SPADE_train.ipynb` | an attempt at anomaly detection with SPADE on the crops in a MVTec-style folder layout; no outcome is recorded |
| `visualize_patchcore.py` | loads a trained PatchCore model and draws anomaly maps for "good" and "disease" test images; no outcome is recorded |
| `clf_dtst_prepare.ipynb` | cuts classifier crops out of full photos with the detector |
| `train.ipynb` | YOLO training cells for the detector (commented out) and for the YOLO classifier (`fish_clf_m`) on `Dataset_autocropped`, the kind of model that [experiment E6](../../docs/experiments.md#e6-frozen-embeddings-against-the-deployed-yolo-classifier) later compared with frozen embeddings |
| `sssss.py` | the detector training recipe (YOLOv8s from the architecture file, 960 px, 300 epochs, mosaic, copy-paste 0.7) that `experiments/detector/det_train.py` adapts (pretrained start, no copy-paste) |
| `export.ipynb`, `onnx_label.ipynb` | exporting the deployed YOLO classifier to ONNX and inspecting the shapes and outputs of exported models |
| `balance_classes.py` | duplicates files of the small classes in the test folder up to 30 per class (one source of the `dup_` copies that [data.md](../../docs/data.md#classifier-data) describes) |
| `find_models.py` | searches directories for `.pth`, `.ckpt` and `.pt` files |
