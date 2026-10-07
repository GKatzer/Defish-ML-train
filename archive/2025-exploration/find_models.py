import os
from pathlib import Path

def find_model_files(start_path="."):
    """Рекурсивно ищет все .pth и .ckpt файлы"""
    model_extensions = {'.pth', '.ckpt', '.pt'}
    model_files = []
    
    for root, dirs, files in os.walk(start_path):
        for file in files:
            if any(file.endswith(ext) for ext in model_extensions):
                full_path = Path(root) / file
                model_files.append(full_path)
                print(f"Found: {full_path}")
    
    return model_files

# Ищем в нескольких вероятных местах
search_paths = [
    "results",
    ".",
    "..", 
    "/g/Work/Fish-Guard_2_0/ML/patchcore-inspection",
    "/g/Work/Fish-Guard_2_0/ML",
    "F:\\Models"
]

all_models = []
for path in search_paths:
    if Path(path).exists():
        print(f"\nSearching in: {path}")
        all_models.extend(find_model_files(path))

# Фильтруем по размеру (модели обычно > 1MB)
filtered_models = []
for model_path in all_models:
    try:
        size_mb = model_path.stat().st_size / (1024 * 1024)
        if size_mb > 1:  # Больше 1MB
            filtered_models.append((model_path, size_mb))
            print(f"Model: {model_path} | Size: {size_mb:.1f} MB")
    except:
        continue

print(f"\n=== Found {len(filtered_models)} potential models ===")
for i, (model_path, size_mb) in enumerate(filtered_models, 1):
    print(f"{i}. {model_path} ({size_mb:.1f} MB)")