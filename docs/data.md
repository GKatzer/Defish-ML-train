# Data and labels

What the datasets are, how they were labelled and split, why the splits look the way they do, and the folder layout the scripts expect. **No image is in this repository**: the photographs are third-party (public posts, other datasets) and are not redistributed. Data will be published: <!-- TODO: insert the link to the published data --> The numbers on this page come from the scripts and reports of the repository (`experiments/detector/build_splits.py`, `experiments/clf_frozen_vs_yolo.py`, `experiments/*/README.md`, `experiments/results/report.md`); none could be recomputed here.

- [Overview](#overview)
- [Detector data](#detector-data)
- [Classifier data](#classifier-data)
- [Why the splits are grouped](#why-the-splits-are-grouped)
- [Gold, silver and external data](#gold-silver-and-external-data)
- [Folder layout the scripts expect](#folder-layout-the-scripts-expect)
- [Provenance and licences](#provenance-and-licences)

## Overview

| Data | Used for | Size | Labels |
|---|---|---|---|
| detector pool: `ds_v2`, `hand_label`, `hl_null_tiny` | training and evaluating the detector | 1069 images (553 + 149 + 367) | one class `fish`, YOLO boxes |
| unlabelled Reddit pools | the pseudo-labelling experiment | 1781 photos in the pool that the report quotes | none |
| `Dataset_autocropped` | the disease classifier | 187 unique crops, 138 of them in the seven gold classes | class per crop, checked by a fish-keeper |
| 5 external healthy-tetra crops | a counter-example for the species confound | 5 | `healthy`, not verified by a fish-keeper |
| `panda992/fish_disease_datasets` | probing and a silver-data experiment | 2450 images, 1747 unique, 128 px and 224 px | 7 pond-fish classes, not reliable |

## Detector data

Three hand-labelled sets of one class, `fish`, were pooled by `build_splits.py`:

| Source | What it is | Images |
|---|---|---|
| `ds_v2` | photos from Reddit aquarium posts, mostly small fish | 553 |
| `hand_label` | close-ups of (often sick) fish, larger boxes | 149 |
| `hl_null_tiny` | close-ups and fish-free aquascapes, used as negatives | 367 |

The pool was **re-split** into train 739, valid 165 and test 165 images (the earlier splits of each source leaked, below). The test split holds 292 fish; under 24 px there are 30 of them and between 24 and 64 px 62 (`experiments/detector/README.md`).

## Classifier data

`Dataset_autocropped` is a folder-per-class set of crops of single fish, cut out of the photos by the detector of the time. File names follow the pattern `<source>_crop<N>_jpg.rf.<hash>.jpg`, the form an image-annotation platform's export gives; the part before `_crop<N>` identifies the **source photo** and is what the scripts group by.

- **Classes.** Nine folders: `healthy` and six diseases (`dermatomycosis`, `fin_rot`, `hexamitosis`, `mycobacteriosis`, `oodiniosis`, `plistophorosis`), plus two helper classes, `many_fish` and `not_a_fish`, that the earlier classifier used as escape classes. The final classifier drops the two helper classes: **seven classes**.
- **Labels.** The crops were labelled and checked by a fish-keeper (the READMEs call them "expert-checked" and "gold").
- **Duplicates.** The export contained `dup_<n>_` copies of crops, and the same images were in `valid` and in `test`. `clf_frozen_vs_yolo.py` rebuilds the set without them: duplicates are found by MD5, a file keeps one record, and `valid` and `test` count as one pool. The result is **187 unique crops** (128 from the training folder, 59 from the evaluation folders), of which **138 belong to the seven gold classes** (`manifest_unique_crops.csv`, written by the script and not committed). The survey and the docstring of `clf_frozen_vs_yolo.py` say 189 where the report says 187; the two were not reconciled.
- **Class sizes** quoted in the reports: 42 `healthy` and 6 `plistophorosis` crops among the 138; the other classes are between 13 and 24 (the confusion matrix in the report has the row sums 13, 24, 42, 17, 21, 15 and 6).

## Why the splits are grouped

Photos of the same tank, crops of the same fish and re-posted pictures look alike; a random split puts near-copies on both sides and inflates every metric.

- **Detector** (`build_splits.py`): images are merged into one **group** when they share a Reddit post id (the first seven characters of the file name, `ds_v2` only), an identical MD5, or a near-identical picture (dHash Hamming distance of at most 6). Whole groups go to one split; per source, 15 % of the images go to test and 15 % to valid, with seed 0.
  The check on the deployed model showed why this matters: its AP50 was 0.74 on images from its own old training split and 0.41 on images it had not seen.
- **Classifier**: crops are grouped by their source photo and evaluated with stratified, grouped k-fold (`StratifiedGroupKFold`, 4 folds, several seeds), so a fish never has crops on both sides of a fold.
- **Pseudo-labelling** (`pseudo_label.py`) applies the same rule against the evaluation splits: pool images that share a post id with a valid or test image, or look like one (dHash distance of at most 6), are excluded.

## Gold, silver and external data

- **Gold**: the 138 seven-class crops checked by a fish-keeper. Every reported estimate uses gold only.
- **Silver**: images that were not checked to the same standard. Five crops of visibly healthy neon tetras were cut with the project's own detector from two Wikimedia Commons photos (CC BY-SA, credited in `train_final.py`) and added to `healthy`. They are always in the final training set and never held out by the cross-validation; they were added because **all six `plistophorosis` crops were neon tetras and no healthy gold crop was**, so "is a neon tetra" and "has plistophorosis" were perfectly correlated ([classifier README](../experiments/classifier/README.md)).
- **External**: `panda992/fish_disease_datasets` (Hugging Face) was downloaded, deduplicated and probed. It is pond and market fish at 128 px, with duplicates inside the set (163 images in both its train and test parts), augmentations baked in and one image under two labels; the classifier does not transfer to it and adding it did not improve the gold estimate ([experiments](experiments.md#e12-external-silver-data-as-training-data), [survey](../experiments/open_datasets_survey.md)).

## Folder layout the scripts expect

Everything hangs off `DEFISH_DATA_DIR` ([paths.py](../experiments/paths.py) resolves it; `python experiments/paths.py` prints the result):

```text
$DEFISH_DATA_DIR/
  Dataset_autocropped/{train,valid,test}/<class>/*.jpg      classifier crops (9 class folders)
  Dataset_hand_label/{train,valid,test}/{images,labels}      detector set (YOLO txt, class 0)
  Dataset_hl_null_tiny-fish/{train,valid,test}/{images,labels}
  Reddit-fishes/aquarium_new/ds_v2/{train,valid,test}/{images,labels}
  Reddit-fishes/{aquarium_new/gallery-dl, aquarium_top_1, aquarium_top_2}/   unlabelled pools
  detector/v3/{train,valid,test}/{images,labels}, manifest.csv, data_*.yaml   written by build_splits.py
  detector/v3_pseudo/                                        written by pseudo_label.py
  external/panda992_fish_disease_datasets/{images/, manifest.csv}
```

## Provenance and licences

- The Reddit photographs are public aquarium posts collected with a downloader; they belong to their authors and are not in this repository.
- The two Wikimedia photographs behind the healthy-tetra crops are CC BY-SA; the crops are not in this repository either.
- `panda992/fish_disease_datasets` states no licence; the survey notes a Kaggle original and a copy with a non-commercial licence (`Saon110`), which must not be used if the application may become commercial.
- Weights trained here carry the licences of what they were built with: the detector was trained with Ultralytics YOLOv8 (AGPL-3.0, noted in the exported file's metadata); the embedder is DINOv2 (`facebook/dinov2-base`).
