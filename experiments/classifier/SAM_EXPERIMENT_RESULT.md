# SAM-masked crops vs rectangular crops: result (2026-09-22)

**Verdict: SAM masking makes the classifier worse. Not adopted.**

Box-prompted `facebook/sam-vit-base` (not automatic/"segment everything" mode) on all 138 gold crops, background
outside the mask flattened to the same neutral colour used elsewhere (124,116,104), same grouped CV as `train_final.py`:

| input | accuracy | fin_rot recall |
|---|---|---|
| rectangular crop (current, deployed) | 0.572 ± 0.036 | 0.56 |
| SAM-masked crop | 0.450 ± 0.028 | 0.35 |

## Why
- **Fin clipping confirmed.** `fin_rot` recall dropped the most of any class -- fin edges are thin, low-contrast,
  and exactly what a promptable segmenter tends to trim, but they're the diagnostic signal for this disease.
- **Mask failures happen and aren't reliably self-flagged.** The area-based sanity filter (reject masks covering
  <15% or >97% of the crop) accepted 138/138 masks, but a manual look at a 12-image sample caught at least one
  clearly inverted mask (background kept, fish erased) that the area heuristic didn't catch, because a wrong mask
  can still have a "plausible" area fraction. A production filter would need something smarter than area alone
  (e.g. cross-check against the detector's own box), which adds complexity for a change that already looks net negative.
- Crops are already tight (they came from a detector box), so there was little background left to remove; the mask
  mostly traded a bit of clean background for a chance of cutting into the fish.

## Files
`sam_crop_experiment.py` (the experiment), `sam_masked_crops/` (138 generated images, for inspection),
`sam_manifest.csv` (per-image mask fraction/IoU/kept-or-fallback).

## Relation to the segmentation-vs-detection question
This was the cheap way to test "would a cleaner (masked) input help the classifier" without training a
segmentation model. It didn't. Combined with the earlier reasoning (detection recall on small fish is data-limited,
not representation-limited, and there is no fish instance-segmentation dataset to train on), there is no evidence
right now that segmentation -- in any form, zero-shot or trained -- would improve either the detector or the
classifier. Not pursuing it further unless something changes (e.g. a source of masks that reliably preserves fin
detail).
