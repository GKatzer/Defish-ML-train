"""Standalone inference for the DINOv2 + logistic-regression classifier.

Mirrors the shape of the server's inference/classifier.py (predict(image) -> {"label", "confidence"}),
extended with "uncertain" and "top3". Not wired into the server — this is for evaluation / manual testing.
Needs torch + transformers + joblib (see requirements.txt).
"""
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel

ART = Path(__file__).parent / "artifacts"
PAD_RGB = (124, 116, 104)  # ImageNet mean colour, matches training preprocessing


class FishDiseaseClassifier:
    def __init__(self, artifacts_dir=ART, device=None):
        meta = json.loads((Path(artifacts_dir) / "meta.json").read_text())
        self.classes = meta["classes"]
        self.threshold = meta["confidence_threshold"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = joblib.load(Path(artifacts_dir) / "scaler.pkl")
        self.clf = joblib.load(Path(artifacts_dir) / "classifier.pkl")
        model_name = {"dinov2_base_224": "facebook/dinov2-base"}[meta["embedder"]]
        self.size = 224
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def _pad_square(self, im: Image.Image) -> Image.Image:
        w, h = im.size
        s = max(w, h)
        canvas = Image.new("RGB", (s, s), PAD_RGB)
        canvas.paste(im, ((s - w) // 2, (s - h) // 2))
        return canvas

    @torch.no_grad()
    def _embed(self, im: Image.Image) -> np.ndarray:
        im = self._pad_square(im.convert("RGB")).resize((self.size, self.size), Image.BICUBIC)
        x = torch.from_numpy(np.array(im)).permute(2, 0, 1)[None].float().to(self.device) / 255
        x = (x - self._mean) / self._std
        feats = []
        for xi in (x, x.flip(-1)):  # hflip test-time augmentation, matches training
            h = self.model(pixel_values=xi).last_hidden_state
            feats.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1))
        return torch.stack(feats).mean(0)[0].cpu().numpy()

    def predict(self, image) -> dict:
        """image: PIL.Image (RGB) or a BGR numpy array (as cv2.imread / crop_by_bbox gives)."""
        if not isinstance(image, Image.Image):
            import cv2
            image = Image.fromarray(image[:, :, ::-1])  # BGR -> RGB
        emb = self._embed(image)[None]
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        proba = self.clf.predict_proba(self.scaler.transform(emb))[0]
        order = np.argsort(-proba)
        top1 = int(order[0])
        return {
            "label": self.classes[top1],
            "confidence": float(proba[top1]),
            "uncertain": bool(proba[top1] < self.threshold),
            "top3": [{"label": self.classes[i], "confidence": float(proba[i])} for i in order[:3]],
        }


if __name__ == "__main__":
    import sys
    clf = FishDiseaseClassifier()
    for path in sys.argv[1:]:
        print(path, "->", clf.predict(Image.open(path)))
