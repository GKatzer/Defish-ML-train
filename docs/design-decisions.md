# Design decisions

Why the project is evaluated and built the way it is, what was chosen over what, and what is still open. The experiments behind each decision, with their numbers, are in [experiments.md](experiments.md); the data are described in [data.md](data.md).

- [1. Evaluate on groups, never on random splits](#1-evaluate-on-groups-never-on-random-splits)
- [2. Evaluate the detector through the production pipeline](#2-evaluate-the-detector-through-the-production-pipeline)
- [3. A frozen embedder and a linear head](#3-a-frozen-embedder-and-a-linear-head)
- [4. Report cross-validation, ship a model fitted on everything](#4-report-cross-validation-ship-a-model-fitted-on-everything)
- [5. Seven classes and an uncertain flag](#5-seven-classes-and-an-uncertain-flag)
- [6. A gate chosen from out-of-fold probabilities](#6-a-gate-chosen-from-out-of-fold-probabilities)
- [7. Silver data only if it helps the gold estimate](#7-silver-data-only-if-it-helps-the-gold-estimate)
- [8. Export so that serving needs no PyTorch](#8-export-so-that-serving-needs-no-pytorch)
- [9. Configure paths with environment variables](#9-configure-paths-with-environment-variables)
- [Open questions](#open-questions)

## 1. Evaluate on groups, never on random splits

**Chosen.** Images that share a post, are byte-identical or look alike (dHash distance of at most 6) are merged into groups, and whole groups go to one split. Crops are grouped by their source photo.
**Why.** The detector that was in production scored AP50 0.74 on images from its own old training split and 0.41 on images it had not seen: the old evaluation measured memory, not skill. The classifier set had `dup_` copies and identical images in `valid` and `test`, which inflates a metric the same way ([E1](experiments.md#e1-the-deployed-detector-on-a-leak-free-split), [data.md](data.md#why-the-splits-are-grouped)).
**Instead of.** The splits that came with each dataset.
**Cost.** The test sets became small (165 images and 292 fish for the detector, 138 crops for the classifier), so a few points of difference are noise and the reports say so.

## 2. Evaluate the detector through the production pipeline

**Chosen.** `det_eval.py` runs candidate detectors through the inference service's own code (`YOLO_ONNX_Inference`, letterbox to 960 px with white padding, confidence 0.25, NMS IoU 0.45), and offers `prod-gray` (the padding colour used in training) and `ultra` (Ultralytics' own predict) to see how much the pipeline itself matters.
**Why.** A number measured with a different preprocessing than the server's is not the number a user gets. Using the service's code, not a copy, removes one source of drift.
**Consequence.** The evaluation depends on a checkout of the service (`DEFISH_INFERENCE_DIR`).

## 3. A frozen embedder and a linear head

**Chosen.** Frozen DINOv2-base embeddings (CLS and mean-patch tokens, averaged over the crop and its mirror image) and a logistic regression with a standard scaler.
**Why.** 138 crops of seven classes are too few to train a network from scratch, and a linear head on a strong general embedding has few parameters to overfit and can be evaluated with grouped cross-validation. The deployed classifier, trained from scratch, scored 0.38 and the frozen embedding 0.51 on the same 45 unseen crops (the intervals overlap: [E6](experiments.md#e6-frozen-embeddings-against-the-deployed-yolo-classifier)); 224 px and 448 px tie and BioCLIP is worse ([E7](experiments.md#e7-which-embedder)), so the smaller input was kept.
**Not adopted.** SAM-masked crops ([E11](experiments.md#e11-sam-masked-crops)), BioCLIP, a larger input, external data.

## 4. Report cross-validation, ship a model fitted on everything

**Chosen.** The expected accuracy comes from grouped, stratified cross-validation on the gold crops (4 folds × 8 seeds); the shipped model is then fitted on all crops (and the five silver healthy-tetra crops) with no holdout.
**Why.** Gold data are too scarce to sacrifice any: the cross-validation estimate is what that final fit is expected to do on new photos.
**Cost.** No independent test set exists for the shipped model; the final fit itself is not evaluated.

## 5. Seven classes and an uncertain flag

**Chosen.** `healthy` plus six diseases; `many_fish` and `not_a_fish` are dropped as predicted classes, and a weak prediction is returned with `uncertain: true` and its top three classes instead of being forced or discarded.
**Why.** The helper classes took training signal from the real ones, and the server discarded them anyway, so a bad crop silently produced no answer. The caller decides what to show for an uncertain prediction.

## 6. A gate chosen from out-of-fold probabilities

**Chosen.** The threshold is the smallest one whose accuracy among the kept predictions reaches a target (0.75), from out-of-fold probabilities pooled over seeds.
**Why.** With 0.57 accuracy, always answering is a coin flip dressed as a finding; the gate trades coverage for trust (0.83 keeps 49 % of crops).
**Limits.** The threshold is selected on the same predictions that give the 49 % and the 0.75, so those are a tuning target; the target itself ("would you trust this in an app") was not validated against a product requirement ([E10](experiments.md#e10-the-confidence-gate)).

## 7. Silver data only if it helps the gold estimate

**Chosen.** External or unverified data enter training only if they raise the gold cross-validation estimate; otherwise they are dropped.
**Why.** The external set looked large (2450 images) and was mostly duplicates, baked-in augmentations and a different domain; a rule fixed in advance prevents "more data" from being accepted on faith. It did not help ([E12](experiments.md#e12-external-silver-data-as-training-data)). The exception is five healthy-tetra crops, added as a targeted counter-example for the species confound ([E9](experiments.md#e9-the-species-confound)) and checked on their own by leave-one-out.

## 8. Export so that serving needs no PyTorch

**Chosen.** The embedder is exported to ONNX with the image normalisation and the pooling inside the graph, and the scaler and the head become plain arrays, so the service needs only `onnxruntime` and NumPy. The export is checked against the PyTorch model (cosine similarity above 0.999).
**Cost.** `export_onnx.py`, `train_final.py` and `finalize_onnx_artifacts.py` must be re-run together after every retraining, and the 348 MB embedder is shipped separately from the code.
**Found later, when the service ran with the real files.** Parity with PyTorch is only as good as the preprocessing: the service once resized crops with OpenCV while `train_final.py` and `infer_onnx.py` use Pillow's bicubic filter, which moved confidences by up to 0.12 on 86 real crops until the service was changed ([the service's decision 6](https://github.com/GKatzer/Defish-inference/blob/master/docs/design-decisions.md#6-onnx-runtime-and-numpy-only-and-the-same-preprocessing-as-training)). Any other runtime of the embedder must resize with Pillow's `Image.BICUBIC`.

## 9. Configure paths with environment variables

**Chosen.** One module, [`experiments/paths.py`](../experiments/paths.py), reads `DEFISH_DATA_DIR`, `DEFISH_RUNS_DIR`, `DEFISH_MODELS_DIR`, `DEFISH_INFERENCE_DIR` and `DEFISH_DEPLOYED_CLASSIFIER`; every script takes its locations from it. Before, the scripts held the author's drive letters (`F:\`, `G:\`, `I:\`).
**Why.** The data are not in the repository, so a reader has to point the scripts at their own copy; one place to set it beats editing constants in seven files.
**Verified how.** The module resolves and prints the paths; every script compiles, and the scripts with a command line print their help. They **could not be run on data** here ([reproducing.md](reproducing.md)).

## Open questions

- More labelled small fish for the detector (recall under 64 px is about 0.3 to 0.45) and more varied healthy-tetra photos for the classifier (one external crop is still called `plistophorosis`).
- Whether the other diseases have a species confound too (not checked).
- A requirement-based choice of the gate's target precision.
- fp16 or int8 for the 348 MB embedder, and whether dropping the mirror-image pass costs accuracy (unmeasured).
