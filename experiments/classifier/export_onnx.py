"""Export the DINOv2 embedder to ONNX so the classifier needs only onnxruntime at inference (no torch).

The exported graph does CLS+mean-patch pooling INSIDE the graph (output: [batch, 1536] float32), so the runtime
side only has to: normalize the image, run the graph twice (original + hflip), average, L2-normalize, then apply
the (tiny, hand-written-in-numpy) scaler + logistic regression -- see infer_onnx.py.

Needs torch + transformers (the shared venv). Run once, on this machine; ship only the .onnx + artifacts to the
target machine, which needs just onnxruntime, numpy and Pillow/opencv.

    python experiments/classifier/export_onnx.py
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from transformers import AutoModel

ART = Path(__file__).parent / "artifacts"
MODEL_NAME = "facebook/dinov2-base"
SIZE = 224
PAD_RGB = (124, 116, 104)


class EmbedderWithPooling(torch.nn.Module):
    """Wraps AutoModel so the ONNX graph outputs the pooled [CLS, mean(patches)] embedding directly."""
    def __init__(self, name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(name)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, pixel_values_0_1):
        x = (pixel_values_0_1 - self.mean) / self.std
        h = self.backbone(pixel_values=x).last_hidden_state
        return torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1)


def pad_square(im: Image.Image) -> Image.Image:
    w, h = im.size
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), PAD_RGB)
    canvas.paste(im, ((s - w) // 2, (s - h) // 2))
    return canvas


def main():
    model = EmbedderWithPooling(MODEL_NAME).eval()
    dummy = torch.rand(1, 3, SIZE, SIZE)  # values in [0, 1], as the runtime side will provide
    onnx_path = ART / "embedder_dinov2_base.onnx"
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["pixel_values"], output_names=["embedding"],
        dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17, do_constant_folding=True,
    )
    size_mb = onnx_path.stat().st_size / 1e6
    print(f"exported {onnx_path.name}  ({size_mb:.0f} MB)")

    # ---- parity check: torch vs onnxruntime, on a few real gold crops
    import pandas as pd
    m = pd.read_csv(Path(__file__).parent.parent / "manifest_unique_crops.csv")
    sample = m.sample(6, random_state=0).path.tolist()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def torch_embed(im):
        im = pad_square(im.convert("RGB")).resize((SIZE, SIZE), Image.BICUBIC)
        x = torch.from_numpy(np.array(im)).permute(2, 0, 1)[None].float() / 255
        with torch.no_grad():
            f = [model(xi) for xi in (x, x.flip(-1))]
        return torch.stack(f).mean(0)[0].numpy()

    def onnx_embed(im):
        im = pad_square(im.convert("RGB")).resize((SIZE, SIZE), Image.BICUBIC)
        x = np.asarray(im).transpose(2, 0, 1)[None].astype(np.float32) / 255
        f = [sess.run(None, {"pixel_values": xi})[0][0] for xi in (x, x[:, :, :, ::-1])]
        return np.mean(f, axis=0)

    diffs = []
    for p in sample:
        im = Image.open(p)
        a, b = torch_embed(im), onnx_embed(im)
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        diffs.append(cos)
        print(f"  {Path(p).name[:50]:50s} cosine(torch, onnx) = {cos:.6f}")
    print(f"min cosine similarity over {len(sample)} samples: {min(diffs):.6f} (should be ~1.0)")
    assert min(diffs) > 0.999, "ONNX export does not match the torch model closely enough"

    meta_path = ART / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["onnx_embedder"] = onnx_path.name
    meta["onnx_embedder_size_mb"] = round(size_mb, 1)
    meta_path.write_text(json.dumps(meta, indent=2))
    print("parity OK, meta.json updated")


if __name__ == "__main__":
    main()
