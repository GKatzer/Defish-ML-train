"""Where the experiment scripts find their data and weights. Set the environment variables, nothing else needs editing.

The module only reads the environment; it creates no directory. `python experiments/paths.py` prints what it resolved to.

DEFISH_DATA_DIR              root of the datasets (default: ./data, relative to the working directory)
    Dataset_autocropped/{train,valid,test}/<class>/*.jpg     classifier crops, one folder per class
    Dataset_hand_label/, Dataset_hl_null_tiny-fish/          hand-labelled detector sets (images + YOLO txt labels)
    Reddit-fishes/aquarium_new/ds_v2/                        the detector's Reddit set
    Reddit-fishes/{aquarium_new/gallery-dl,aquarium_top_1,aquarium_top_2}/   unlabelled pools (pseudo_label.py)
    detector/v3/                                             written by build_splits.py
    detector/v3_pseudo/                                      written by pseudo_label.py
    external/panda992_fish_disease_datasets/{images/,manifest.csv}   external silver data
DEFISH_RUNS_DIR              training runs of det_train.py (default: ./runs)
DEFISH_MODELS_DIR            weights of the baselines compared by det_eval.py: detector.onnx, new_colab.onnx, nout.pt,
                             yolo.pt (default: ./models); det_train.py also runs from this directory
DEFISH_INFERENCE_DIR         a checkout of the inference service (Defish-inference); det_eval.py imports its
                             inference.detector and utils.img to evaluate under the production pipeline, and takes
                             its models/best.onnx as the "PROD" baseline (default: ../Defish-inference)
DEFISH_DEPLOYED_CLASSIFIER   the old YOLO classifier fish_classifier.pt, only for the baseline of clf_frozen_vs_yolo.py
                             (default: ./models/fish_classifier.pt)
HF_HOME                      Hugging Face cache; the scripts do not set it (set it if the default disk is too small)
"""
import os
from pathlib import Path


def _path(variable: str, default: str) -> Path:
    return Path(os.environ.get(variable, default)).expanduser()


DATA_DIR = _path("DEFISH_DATA_DIR", "data")
RUNS_DIR = _path("DEFISH_RUNS_DIR", "runs")
MODELS_DIR = _path("DEFISH_MODELS_DIR", "models")
INFERENCE_DIR = _path("DEFISH_INFERENCE_DIR", "../Defish-inference")
DEPLOYED_CLASSIFIER = _path("DEFISH_DEPLOYED_CLASSIFIER", str(Path("models") / "fish_classifier.pt"))

CLASSIFIER_CROPS = DATA_DIR / "Dataset_autocropped"
DETECTOR_V3 = DATA_DIR / "detector" / "v3"
DETECTOR_V3_PSEUDO = DATA_DIR / "detector" / "v3_pseudo"
REDDIT_ROOT = DATA_DIR / "Reddit-fishes"
EXTERNAL_PANDA992 = DATA_DIR / "external" / "panda992_fish_disease_datasets"

if __name__ == "__main__":
    for name in ("DATA_DIR", "RUNS_DIR", "MODELS_DIR", "INFERENCE_DIR", "DEPLOYED_CLASSIFIER", "CLASSIFIER_CROPS",
                 "DETECTOR_V3", "DETECTOR_V3_PSEUDO", "REDDIT_ROOT", "EXTERNAL_PANDA992"):
        value = globals()[name]
        print(f"{name:22} {value}  ({'exists' if value.exists() else 'missing'})")
