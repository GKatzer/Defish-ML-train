import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from patchcore.patchcore import PatchCore
import patchcore.common

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Загрузка модели ===
model = PatchCore(device)
nn_method = patchcore.common.FaissNN(on_gpu=False, num_workers=4)
model.load_from_path(
    r"G:\Work\Fish-Guard_2_0\ML\patchcore-inspection\results\project\group_4\models\mvtec_fish",
    device=device,
    nn_method=nn_method,
)

# === Предобработка (как в датасете) ===
def preprocess_for_patchcore(image_path, resize=256, imagesize=224):
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (resize, resize), interpolation=cv2.INTER_AREA)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image.astype(np.float32) / 255.0 - mean) / std

    # Финальный ресайз (все приводим к 224x224)
    image = cv2.resize(image, (imagesize, imagesize), interpolation=cv2.INTER_AREA)

    # HWC -> CHW + batch dim → torch.Tensor
    img_torch = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

    return img_torch



good_scores = []
bad_scores = []

for img_path in Path(r"F:\Datasets\Reddit-fishes\aquarium_new\mvtec\fish\test\good").glob("*.jpg"):
    img_torch = preprocess_for_patchcore(img_path).to(torch.float).to(device)
    scores, _ = model._predict(img_torch)
    good_scores.append(scores[0].item())

for img_path in Path(r"F:\Datasets\Reddit-fishes\aquarium_new\mvtec\fish\test\desease").glob("*.jpg"):
    img_torch = preprocess_for_patchcore(img_path).to(torch.float).to(device)
    scores, _ = model._predict(img_torch)
    bad_scores.append(scores[0].item())

print("Средний good:", np.mean(good_scores))
print("Средний disease:", np.mean(bad_scores))

import matplotlib.pyplot as plt

plt.hist(good_scores, bins=30, alpha=0.5, label="good")
plt.hist(bad_scores, bins=30, alpha=0.5, label="disease")
plt.legend()
plt.show()

# (the list of test images that the commented-out loop below iterated over was removed from this copy)
# for image_path in test_images:
#     if not Path(image_path).exists():
#         print("Нет файла:", image_path)
#         continue

#     # preprocess -> HWC float32
#     img = preprocess_for_patchcore(image_path)

#     # Вызов предсказания
#     img_torch = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
#     img_torch = img_torch.to(torch.float).to(device)

#     scores, segmentations = model._predict(img_torch)

#     anomaly_score = scores[0]
#     anomaly_map = segmentations[0]

#     # Визуализация
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
#     original = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)

#     ax1.imshow(original)
#     ax1.set_title(f"Original\nScore={anomaly_score:.3f}")
#     ax1.axis("off")

#     im = ax2.imshow(anomaly_map, cmap="jet")
#     ax2.set_title("Anomaly Map")
#     ax2.axis("off")
#     plt.colorbar(im, ax=ax2)

#     plt.show()
