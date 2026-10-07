"""Writes a tiny SYNTHETIC detector dataset in the layout the scripts expect, so that the plumbing can be checked without
the private data: `build_splits.py` (grouping and splitting) and `det_eval.py` (evaluation through the production
pipeline) can be run on it end to end. The images are random noise with a few drawn rectangles as "fish"; the numbers
that come out of an evaluation on them mean nothing, and no claim about the models can be made from them.

usage: DEFISH_DATA_DIR=/tmp/defish-data python docs/examples/synthetic_detector_data.py [--standin-detector FILE.onnx]
With --standin-detector it also writes an ONNX "detector" whose output is a constant (two fixed boxes, YOLOv8 output layout
(1, 5, 2), 960 px frame), so that `det_eval.py --backend prod-white --model FILE.onnx` has something to load; it detects
nothing real.
Needs numpy and Pillow (and onnx for --standin-detector). Fixed seed. Includes, on purpose, byte-identical copies and a shared "Reddit post id", so that the
group logic has something to merge.
"""
import argparse
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

root = Path(os.environ.get("DEFISH_DATA_DIR", "data"))
rng = np.random.default_rng(0)


def make_image(path: Path, label_path: Path, n_fish: int, size=(320, 240)):
    arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    lines = []
    for _ in range(n_fish):
        w, h = int(rng.integers(20, 80)), int(rng.integers(12, 40))
        x, y = int(rng.integers(0, size[0] - w)), int(rng.integers(0, size[1] - h))
        draw.rectangle([x, y, x + w, y + h], fill=(255, 140, 0))
        lines.append(f"0 {(x + w / 2) / size[0]:.6f} {(y + h / 2) / size[1]:.6f} {w / size[0]:.6f} {h / size[1]:.6f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=92)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


sources = {  # name -> (directory, images per old split, fish per image, file-name prefix)
    "ds_v2": (root / "Reddit-fishes" / "aquarium_new" / "ds_v2", {"train": 40, "valid": 6, "test": 6}, (2, 6), "post"),
    "hand_label": (root / "Dataset_hand_label", {"train": 14, "valid": 3, "test": 3}, (1, 2), "close"),
    "hl_null_tiny": (root / "Dataset_hl_null_tiny-fish", {"train": 14, "valid": 3, "test": 3}, (0, 1), "null"),
}
for name, (folder, counts, fish, prefix) in sources.items():
    for split, n in counts.items():
        for i in range(n):
            stem = f"{prefix}{i:03d}" if name != "ds_v2" else f"{prefix}{i // 2:03d}x-{split}-{i}"  # two images per fake post id
            make_image(folder / split / "images" / f"{stem}.jpg", folder / split / "labels" / f"{stem}.txt", int(rng.integers(*fish)))

# byte-identical copies in different old splits: they must end up in the same new split
ds = sources["ds_v2"][0]
src_img = next((ds / "train" / "images").iterdir())
for k, split in enumerate(("valid", "test")):
    shutil.copy2(src_img, ds / split / "images" / f"copy{k}-of-{src_img.name}")
    shutil.copy2(ds / "train" / "labels" / f"{src_img.stem}.txt", ds / split / "labels" / f"copy{k}-of-{src_img.stem}.txt")
print("synthetic detector data written to", root)


parser = argparse.ArgumentParser()
parser.add_argument("--standin-detector", metavar="FILE")
args = parser.parse_args()
if args.standin_detector:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    boxes = np.array([(300, 300, 160, 90, 0.88), (620, 420, 120, 70, 0.74)], dtype=np.float32).T[None, ...]  # (1, 5, 2)
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["output0"], value=numpy_helper.from_array(boxes, "boxes"))], "standin_detector",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, ["batch", 3, "height", "width"])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 5, 2])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    onnx.checker.check_model(model)
    Path(args.standin_detector).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.standin_detector)
    print("stand-in detector written to", args.standin_detector)
