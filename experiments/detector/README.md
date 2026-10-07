# Fish detector: leakage-free evaluation and a better model (2026-09)

## What was done
1. `build_splits.py` pools the three hand-labelled sets (ds_v2 = Reddit aquariums, hand_label and hl_null_tiny = close-ups
   and fish-free aquascapes) into 1069 images and splits them by group (same Reddit post, identical or near-identical
   picture), so the same tank never sits in train and test. Output: `$DEFISH_DATA_DIR/detector/v3` (train 739 / valid 165 / test 165).
2. `det_eval.py` evaluates any detector under the **production pipeline** of the inference service (`Defish-inference`, found through `DEFISH_INFERENCE_DIR`)
   (`letterbox_resize` to 960 + `YOLO_ONNX_Inference`, conf 0.25 / IoU 0.45) and reports AP50, precision/recall,
   FP per image, and recall by fish size. Predictions are cached in `cache/`, results in `results/det_results.csv`.
3. `det_train.py` trains from pretrained `yolov8s.pt` (imgsz 960, mosaic, scale 0.5, hsv; `cache=False`, see below).
4. `pseudo_label.py` tried self-training (see "Did not help").

## Result (test split, 165 images / 292 fish, production pipeline, conf 0.25)
| Model | AP50 | Precision | Recall | FP/img |
|---|---|---|---|---|
| Deployed `best.onnx` | 0.55 | 0.69 | 0.55 | 0.44 |
| yolov8s pretrained, train only | 0.66 | 0.69 | 0.64 | 0.50 |
| **yolov8s pretrained, train+valid (exported ONNX)** | **0.66** | 0.73 | 0.62 | 0.40 |

The deployed model memorised part of the Reddit images (AP50 0.74 on images from its old train split, 0.41 on unseen
ones), so the fair comparison drops those: on the 117 leak-free test images, AP50 0.72 vs 0.42 (bootstrap 95% CI of the
difference +0.23..+0.37) and recall 0.70 vs 0.44. The biggest gain is on close-ups of sick fish (AP50 0.51 -> 0.92 and
0.44 -> 0.75), a domain the deployed model never saw. CPU speed is identical (~200 ms per 960x960 frame).

Adding valid to train gave no measurable gain on test (0.66 -> 0.66), so more of the same data is not the lever; more
labelled **small** fish is (recall on fish under 64 px is still only ~0.3-0.45).

## Did not help
- imgsz 1280 (train and inference): AP50 0.65, no gain, ~1.8x cost. Inference-only 1280 on a 960 model hurts (0.58).
- Tiled inference (full frame + overlapping tiles): tiny-fish recall 0.43 -> 0.60 but FP/img 0.5 -> 1.1, AP50 unchanged.
- Self-training with two-model agreement on 1781 unlabeled Reddit photos: only 199 accepted, median fish 243 px (easy,
  large fish), so it cannot help with small ones. Not trained.

## Caveats
- Test is small (tiny <24 px: 30 fish, 24-64 px: 62). Differences of a few points between runs are noise.
- ds_v2 empty images are real negatives (turtles, snails, test kits), the label audit found only a few missing or loose boxes.
- Training with `cache=True` crashes on Windows (pickling the RAM cache to DataLoader workers) once the set exceeds ~2 GB.

## Deploying (not done automatically)
`$DEFISH_RUNS_DIR/fish_det/export/fish_detector_v8s_pretrained_trainval.onnx` (the path of the run on the author's machine) is a drop-in replacement for `models/best.onnx`
(dynamic input `[batch,3,h,w]`, output `[batch,5,anchors]`, verified with the server's own `detector.py`). No code change is needed.
