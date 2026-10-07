# Experiments and their verdicts

Every experiment of the project in one place: the question, how it was run, the result with its sample size, and the verdict. Positive and negative results are listed alike. The numbers are those of the reports in this repository ([detector](../experiments/detector/README.md), [classifier](../experiments/classifier/README.md), [SAM](../experiments/classifier/SAM_EXPERIMENT_RESULT.md), [report.md](../experiments/results/report.md), [survey](../experiments/open_datasets_survey.md)); they were not recomputed for this page because the data are not in the repository. Intervals are the 95 % intervals the reports give; "sd" is a standard deviation over seeds.

| # | Question | Verdict |
|---|---|---|
| [E1](#e1-the-deployed-detector-on-a-leak-free-split) | Is the deployed detector as good as it looked? | **No**: it had memorised its data (AP50 0.41 on unseen images) |
| [E2](#e2-pretrained-yolov8s-on-train-or-train--valid) | Does a pretrained YOLOv8s beat it, and does adding the validation split help? | **Yes** (AP50 0.66 against 0.55); the validation split did not help |
| [E3](#e3-resolution-1280-px) | Does 1280 px help? | **No** (AP50 0.65 at 1.8 times the cost) |
| [E4](#e4-tiled-inference) | Do tiles help small fish? | **No** (recall 0.43 → 0.60, false positives doubled, AP50 unchanged) |
| [E5](#e5-self-training-with-two-model-agreement) | Can unlabelled photos be pseudo-labelled to help small fish? | **No**: only easy, large fish pass; not trained |
| [E6](#e6-frozen-embeddings-against-the-deployed-yolo-classifier) | Do frozen embeddings beat the deployed YOLO classifier? | **Probably** (0.51 against 0.38, but the intervals overlap) |
| [E7](#e7-which-embedder) | Which embedder? | **DINOv2-base at 224 px** (448 ties; BioCLIP is worse) |
| [E8](#e8-dropping-the-helper-classes) | Keep `many_fish` and `not_a_fish`? | **No**: seven classes |
| [E9](#e9-the-species-confound) | Is the model recognising the disease or the species? | **Partly the species**; mitigated, not solved |
| [E10](#e10-the-confidence-gate) | Can the model say "not sure"? | **Yes**: gate at 0.83 keeps 49 % of crops at about 75 % |
| [E11](#e11-sam-masked-crops) | Does removing the background with SAM help? | **No** (0.45 against 0.57) |
| [E12](#e12-external-silver-data-as-training-data) | Does external data help? | **No** |
| [E13](#e13-onnx-export-and-parity) | Can it run without PyTorch? | **Yes**; parity above 0.999 |
| [E14](#e14-open-datasets-survey) | Is there a reliable open dataset of aquarium-fish disease? | **No** |
| [E15](#e15-the-2025-exploration) | What did the 2025 exploration try? | anomaly detection and others, no recorded outcome |

## Detector

Evaluation always uses the leak-free split of [data.md](data.md#why-the-splits-are-grouped) and the **production pipeline** (letterbox to 960 px with white padding, the inference service's own `YOLO_ONNX_Inference`, confidence 0.25, NMS IoU 0.45): `det_eval.py` has backends `prod-white`, `prod-gray` and `ultra`, and reports AP50 at IoU 0.5, precision and recall at confidence 0.25, false positives per image, best F1 and recall by fish size. `--baselines` compares six earlier models (the deployed `best.onnx` with white and with grey padding, `detector.onnx`, `new_colab.onnx`, `nout.pt`, `yolo.pt`); the results CSV is not committed, so the tables below are those of the README.

### E1: the deployed detector on a leak-free split
- **Result.** The deployed model's AP50 was **0.74** on images from its old training split and **0.41** on images it had not seen. On the 117 test images it had never seen, AP50 was 0.42 for the deployed model and 0.72 for the new one (95 % bootstrap interval of the difference +0.23 to +0.37; recall 0.44 against 0.70). On the full test split (165 images, 292 fish) the numbers are 0.55 and 0.66 ([table](../README.md#results)).
- **Verdict.** The deployed model had memorised part of its Reddit images; comparisons that include those images flatter it.

### E2: pretrained YOLOv8s on train or train + valid
- **Result** (test split, 165 images and 292 fish): pretrained YOLOv8s on `train` reaches AP50 0.66 (precision 0.69, recall 0.64, 0.50 false positives per image); on `train + valid` AP50 0.66, precision 0.73, recall 0.62, **0.40 false positives per image**. The biggest gain over the deployed model is on close-ups of sick fish (AP50 0.51 → 0.92 and 0.44 → 0.75 on two subsets), a domain the old model never saw.
- **Verdict.** Adopted: `train + valid`, exported to ONNX. More data of the same kind did not move the test AP50; **recall on fish under 64 px stays at about 0.3 to 0.45**, so more labelled *small* fish is the lever (62 test fish between 24 and 64 px and 30 under 24 px: the estimate is noisy).

### E3: resolution 1280 px
- **Result.** Training and inference at 1280 px: AP50 0.65, no gain, about 1.8 times the cost. Inference at 1280 px with a model trained at 960 px: 0.58.
- **Verdict.** Not adopted.

### E4: tiled inference
- **Result.** Full frame plus overlapping tiles: recall on tiny fish 0.43 → 0.60, but false positives per image 0.5 → 1.1 and AP50 unchanged.
- **Verdict.** Not adopted.

### E5: self-training with two-model agreement
- **Method** (`pseudo_label.py`): a box is accepted when the new detector sees it with confidence of at least 0.5 and an independently trained one sees an overlapping box (IoU of at least 0.5) with confidence of at least 0.4; an image is dropped when either model has a strong detection (at least 0.5) that the other does not confirm. Pool images that resemble an evaluation image are excluded.
- **Result.** From 1781 unlabelled Reddit photos only 199 were accepted, with a median fish of 243 px: easy, large fish.
- **Verdict.** It cannot help with small fish; not trained.

## Classifier

All estimates use the gold crops only ([data.md](data.md#gold-silver-and-external-data)) and groups of crops by source photo.

### E6: frozen embeddings against the deployed YOLO classifier
- **Method.** `clf_frozen_vs_yolo.py`, part A: train on the 128 unique crops of the training folder, test on the unique crops of the evaluation folders; frozen embeddings plus a logistic regression against the deployed YOLO classifier (trained from scratch, evaluated with the server's preprocessing and with its native one).
- **Result** (seven classes, 45 evaluation crops): YOLO classifier **0.29** [0.18, 0.43] with the server's preprocessing and **0.38** [0.25, 0.52] with the native one; DINOv2-224 + LR **0.51** [0.37, 0.65]; DINOv2-448 0.53 [0.39, 0.67]; BioCLIP 0.40 [0.27, 0.55]. On the 29 crops whose photo was not in the training set: 0.21 [0.10, 0.38] and 0.38 [0.23, 0.56] for the YOLO classifier, 0.48 [0.31, 0.66] for DINOv2-224.
- **Verdict.** Frozen embeddings are better in the point estimate, and the direction is the same on every subset. The **intervals overlap, so the size of the difference is not established** (0.38 against 0.51 is not "1.5 times better").

### E7: which embedder
- **Result** (part B, stratified group 4-fold cross-validation over the unique crops, sd over seeds): seven classes, 138 crops: DINOv2-224 **0.57** (sd 0.03), DINOv2-448 0.58 (0.03), BioCLIP-224 0.42 (0.02); healthy against sick: 0.81 for both DINOv2 sizes, 0.71 for BioCLIP; nine classes, 187 crops: 0.63, 0.60, 0.52. The shipped model, 4 folds × 8 seeds: **0.572 ± 0.036**, balanced accuracy 0.58, macro F1 0.56, top-3 accuracy **0.84**.
- **Verdict.** DINOv2-base at 224 px: tied with 448 px, smaller and faster. BioCLIP is clearly worse.

### E8: dropping the helper classes
- **Reasoning.** `many_fish` and `not_a_fish` were escape classes: the server already discarded them, they cost the seven real classes two of nine classes' worth of training signal, and a bad detector crop left the caller no fallback. The final classifier has seven classes and an `uncertain` flag instead.
- **Verdict.** Adopted. (The nine-class and seven-class accuracies in E7 are different tasks and are not compared.)

### E9: the species confound
- **Finding.** All six `plistophorosis` crops were neon tetras and no healthy gold crop was, so the model could be recognising the species. Five external healthy-tetra crops were added as silver data and checked on their own by leave-one-out: healthy tetras called `plistophorosis` fell from **3 of 5 to 1 of 5**.
- **Verdict.** Mitigated, not solved: the remaining photo has a different style (macro close-up, grey background), which looks like a photographic confound on top of the species one. The other diseases were not checked for the same failure.

### E10: the confidence gate
- **Method** (`train_final.py`): out-of-fold probabilities pooled over 8 seeds × 4 folds; the threshold is the smallest one, among those that keep at least 20 predictions, whose accuracy among the kept predictions reaches `TARGET_PRECISION = 0.75`.
- **Result** (same predictions): threshold 0.00 keeps 100 % at accuracy 0.57; 0.45 keeps 92 % at 0.60; 0.65 keeps 70 % at 0.67; **0.83 keeps 49 % at 0.75** ([figure](../README.md#results)).
- **Verdict.** Shipped at 0.83. The threshold was selected on the same predictions the table comes from, so 0.75 is a target, not an independent measurement, and 0.75 itself is "a reasonable bar" that was not validated against a product requirement.

### E11: SAM-masked crops
- **Method** (`sam_crop_experiment.py`): box-prompted `facebook/sam-vit-base` on all 138 gold crops, background flattened to the neutral colour (124, 116, 104), same grouped cross-validation.
- **Result.** Accuracy **0.450 ± 0.028** against 0.572 ± 0.036 for rectangular crops; `fin_rot` recall 0.56 → 0.35, because the masks clip thin fins, which are the diagnostic signal. The area filter (reject masks covering under 15 % or over 97 % of the crop) accepted 138 of 138 masks, yet a look at 12 images found at least one inverted mask.
- **Verdict.** Not adopted; no evidence that segmentation, zero-shot or trained, would help either the detector or the classifier.

### E12: external silver data as training data
- **Probe C1** (`panda992`, about 250 images per class, gold-trained DINOv2-224): the model calls only **0.54** of the external `healthy_fish` images healthy, and 0.29 to 0.55 of the diseased ones; the domain (ponds, markets, fish in hands) does not transfer.
- **Experiment C2** (group cross-validation on gold, silver in the training folds only, 138 crops): for DINOv2-224 `gold only` 0.57, `+ healthy ×100` 0.59, `+ healthy ×100 + fungal → dermatomycosis ×100` 0.57; for DINOv2-448 0.58, 0.59, 0.57; for BioCLIP 0.42, 0.40, 0.41.
- **Verdict.** Differences of 0.01 to 0.02 are within the seed noise; the rule that was set beforehand, "if it does not improve the gold estimate, discard it", applies. Not used (apart from the five tetra crops of E9).

### E13: ONNX export and parity
- **Method** (`export_onnx.py`, `finalize_onnx_artifacts.py`, `infer_onnx.py`): the DINOv2 embedder is exported to ONNX with the image normalisation and the pooling (CLS token and mean of the patch tokens, 1536 numbers) inside the graph; the scaler and the logistic regression become plain arrays in `head_numpy.npz`.
- **Result.** Cosine similarity with the PyTorch model above 0.999 on real crops; 348 MB at fp32; about 130 ms per forward pass on a CPU, so about 260 ms per prediction with the mirror image. fp16 or int8 would shrink the file; not done.
- **Verdict.** Adopted: the serving side needs `onnxruntime` and NumPy only ([`Defish-inference`](https://github.com/GKatzer/Defish-inference)).

## Beyond the models

### E14: open datasets survey
A web and Hugging Face survey ([experiments/open_datasets_survey.md](../experiments/open_datasets_survey.md)) found **no reliable open dataset of aquarium-fish disease**: what exists is pond and aquaculture fish, and most "different" datasets are one 7-class set copied across platforms (about 250 images per class, to be deduplicated by hash). For the detector there is far more open data (Open Images, DeepFish, FishNet and others), mostly underwater or fishing scenes. Items that could not be opened (Kaggle, Roboflow returned 403) are marked "not checked" in the survey; their licences must be read before any use.

### E15: the 2025 exploration
Before the experiments above, the 2025 notebooks tried an anomaly-detection route (SPADE and PatchCore on fish crops), auto-labelling with the first detector, and a YOLO classifier trained from scratch (the deployed one that E6 beat). They are kept in [archive/2025-exploration](../archive/2025-exploration/README.md), with local paths and no recorded outcome for the anomaly-detection attempts; they are not part of the reproducible work.
