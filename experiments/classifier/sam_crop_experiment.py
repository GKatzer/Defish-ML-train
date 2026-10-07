"""Does a SAM mask (background flattened) beat the plain rectangular crop for the classifier?

Box-prompted SAM (facebook/sam-vit-base), NOT automatic/"segment everything" mode: for each gold crop, prompt SAM
with a box slightly inset from the crop's own edges (the crop already comes from a detector box, so the fish
should fill most of the frame; a small inset avoids prompting exactly on background-only edge pixels). Take SAM's
highest-IoU-predicted mask. Background (outside mask) is flattened to the same neutral colour already used for
letterbox padding elsewhere (124, 116, 104), so the DINOv2 embedder sees a familiar "padding" colour, not real
background texture, without switching to a 4-channel/alpha input (DINOv2's patch embedding is a fixed 3-channel
conv; RGBA would need retraining it, defeating the point of a frozen embedder).

If a mask covers too little of the crop (SAM latched onto a fin, an eye highlight, a plant leaf) or the whole
crop with no discrimination, the ORIGINAL rectangular crop is kept instead, flagged in the manifest -- a bad mask
should not be worse than no mask.

Evaluated with the exact same grouped stratified CV as train_final.py, most classes AND fin_rot specifically
(the concern: does clipping the mask boundary cut off fin detail that fin_rot diagnosis needs?).

    python experiments/classifier/sam_crop_experiment.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from train_final import DROP, EMBEDDER, cv_out_of_fold, embed_dinov2

HERE = Path(__file__).parent
EXP = HERE.parent
OUT_DIR = HERE / "sam_masked_crops"
OUT_DIR.mkdir(exist_ok=True)
PAD_RGB = (124, 116, 104)
INSET_FRAC = 0.04       # prompt box inset from the crop edges, as a fraction of width/height
MIN_MASK_FRAC = 0.15    # below this fraction of the crop area, distrust the mask (likely a fin/eye fragment)
MAX_MASK_FRAC = 0.97    # above this, the mask is basically "everything" -- no real segmentation happened


def load_sam():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SamModel.from_pretrained("facebook/sam-vit-base").to(device).eval()
    proc = SamProcessor.from_pretrained("facebook/sam-vit-base")
    return model, proc, device


@torch.no_grad()
def sam_mask(model, proc, device, im: Image.Image):
    w, h = im.size
    dx, dy = w * INSET_FRAC, h * INSET_FRAC
    box = [[dx, dy, w - dx, h - dy]]
    inputs = proc(im, input_boxes=[box], return_tensors="pt").to(device)
    out = model(**inputs, multimask_output=True)
    masks = proc.image_processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"].cpu(),
                                                     inputs["reshaped_input_sizes"].cpu())[0][0]  # (3, H, W) bool
    scores = out.iou_scores[0, 0].cpu().numpy()
    best = int(scores.argmax())
    return masks[best].numpy(), float(scores[best])


def apply_mask(im: Image.Image, mask: np.ndarray) -> tuple[Image.Image, float]:
    arr = np.array(im.convert("RGB"))
    frac = float(mask.mean())
    if frac < MIN_MASK_FRAC or frac > MAX_MASK_FRAC:
        return im, frac  # untrustworthy mask: fall back to the original crop, unmasked
    out = arr.copy()
    out[~mask] = PAD_RGB
    return Image.fromarray(out), frac


def build_masked_crops(manifest: pd.DataFrame) -> pd.DataFrame:
    model, proc, device = load_sam()
    rows = []
    for i, r in enumerate(manifest.itertuples()):
        im = Image.open(r.path).convert("RGB")
        mask, iou = sam_mask(model, proc, device, im)
        masked, frac = apply_mask(im, mask)
        used_mask = masked is not im
        out_path = OUT_DIR / f"{Path(r.path).stem}.jpg"
        masked.save(out_path, quality=95)
        rows.append(dict(md5=r.md5, masked_path=str(out_path), mask_iou=iou, mask_frac=frac, used_mask=used_mask))
        if (i + 1) % 40 == 0:
            print(f"  masked {i + 1}/{len(manifest)}", flush=True)
    return pd.DataFrame(rows)


def main():
    manifest = pd.read_csv(EXP / "manifest_unique_crops.csv")
    keep = ~manifest.label.isin(DROP)
    df = manifest[keep].reset_index(drop=True)

    cache = HERE / "sam_manifest.csv"
    if cache.exists():
        sam_df = pd.read_csv(cache)
    else:
        sam_df = build_masked_crops(df)
        sam_df.to_csv(cache, index=False)
    df = df.merge(sam_df, on="md5")
    print(f"masks kept (trusted): {df.used_mask.sum()}/{len(df)}  "
          f"(mean mask fraction of crop area: {df.mask_frac.mean():.2f})")

    emb_cache = HERE / f"sam_masked_{EMBEDDER}.npy"
    if emb_cache.exists():
        Xm = np.load(emb_cache)
    else:
        Xm = embed_dinov2(df.masked_path.tolist())
        np.save(emb_cache, Xm)
    X_rect = np.load(EXP / "cache" / f"gold_{EMBEDDER}.npy")[keep.values]

    y, groups = df.label.values, df.group.values
    print("\ncomparing rectangular crop vs SAM-masked crop, same grouped CV protocol:")
    for name, X in [("rectangular crop (current)", X_rect), ("SAM-masked crop (background flattened)", Xm)]:
        oof = cv_out_of_fold(X, y, groups)
        per_seed = oof.groupby("seed").apply(lambda g: pd.Series(dict(
            acc=(g.true == g.pred).mean(),
            fin_rot_recall=(g[g.true == "fin_rot"].pred == "fin_rot").mean() if (g.true == "fin_rot").any() else np.nan)))
        print(f"  {name:42s} acc={per_seed.acc.mean():.3f}±{per_seed.acc.std():.3f}  "
              f"fin_rot_recall={per_seed.fin_rot_recall.mean():.2f}")


if __name__ == "__main__":
    main()
