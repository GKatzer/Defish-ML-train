"""Pure onnxruntime + numpy inference for the disease classifier. No torch, no scikit-learn at runtime.

Needs: onnxruntime, numpy, Pillow (or swap for cv2, same as the rest of the server). Same shapes/threshold logic
as infer.py (the torch version used for development), so predictions match it (see export_onnx.py's parity check).

    python ML/experiments/classifier/infer_onnx.py path/to/crop.jpg [more crops...]
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ART = Path(__file__).parent / "artifacts"
PAD_RGB = (124, 116, 104)
SIZE = 224


class FishDiseaseClassifierONNX:
    def __init__(self, artifacts_dir=ART):
        artifacts_dir = Path(artifacts_dir)
        meta = json.loads((artifacts_dir / "meta.json").read_text())
        self.classes = meta["classes"]
        self.threshold = meta["confidence_threshold"]
        self.sess = ort.InferenceSession(str(artifacts_dir / meta["onnx_embedder"]), providers=["CPUExecutionProvider"])
        p = np.load(artifacts_dir / "head_numpy.npz")
        self.mean, self.scale = p["scaler_mean"], p["scaler_scale"]       # StandardScaler, applied after L2-normalize
        self.coef, self.intercept = p["lr_coef"], p["lr_intercept"]       # LogisticRegression (multinomial)
        assert list(p["classes"]) == self.classes

    def _pad_square(self, im: Image.Image) -> Image.Image:
        w, h = im.size
        s = max(w, h)
        canvas = Image.new("RGB", (s, s), PAD_RGB)
        canvas.paste(im, ((s - w) // 2, (s - h) // 2))
        return canvas

    def _embed(self, im: Image.Image) -> np.ndarray:
        im = self._pad_square(im.convert("RGB")).resize((SIZE, SIZE), Image.BICUBIC)
        x = np.asarray(im).transpose(2, 0, 1)[None].astype(np.float32) / 255
        feats = [self.sess.run(None, {"pixel_values": xi})[0][0] for xi in (x, x[:, :, :, ::-1])]  # orig + hflip TTA
        return np.mean(feats, axis=0)

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def predict(self, image) -> dict:
        """image: PIL.Image (RGB) or a BGR numpy array (cv2.imread / crop_by_bbox)."""
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image[:, :, ::-1])  # BGR -> RGB
        emb = self._embed(image)
        emb = emb / (np.linalg.norm(emb) + 1e-9)                 # L2-normalize, as in training
        emb = (emb - self.mean) / self.scale                     # StandardScaler
        logits = emb @ self.coef.T + self.intercept
        proba = self._softmax(logits) if logits.ndim == 1 else np.array([self._softmax(z) for z in logits])[0]
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
    clf = FishDiseaseClassifierONNX()
    for path in sys.argv[1:]:
        print(path, "->", clf.predict(Image.open(path)))
