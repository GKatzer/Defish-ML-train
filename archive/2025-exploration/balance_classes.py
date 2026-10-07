import os
import shutil
from pathlib import Path
from random import choices

# === Настройки ===
TARGET_SIZE = 30
ROOT = Path(".\datasets\Dataset_autocropped/test")

# === Цикл по классам ===
for class_dir in ROOT.iterdir():
    if not class_dir.is_dir():
        continue

    images = list(class_dir.glob("*"))
    n_current = len(images)

    if n_current >= TARGET_SIZE:
        print(f"[✓] {class_dir.name}: {n_current} (ok)")
        continue

    # Сколько надо дополнить
    n_needed = TARGET_SIZE - n_current
    to_duplicate = choices(images, k=n_needed)

    # Копируем
    for i, src in enumerate(to_duplicate):
        dst = class_dir / f"dup_{i}_{src.name}"
        shutil.copy2(src, dst)

    print(f"[+] {class_dir.name}: добавлено {n_needed} файлов → {TARGET_SIZE} итогово")
