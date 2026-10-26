# -*- coding: utf-8 -*-
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import time
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm

cache_dir = Path("data/cache_224")
cache_dir.mkdir(parents=True, exist_ok=True)

splits = ["train", "val", "test"]
total_images = 0

t0 = time.time()
for split in splits:
    csv_path = f"data/splits/{split}.csv"
    df = pd.read_csv(csv_path)
    print(f"Cắt tiền xử lý ảnh 224x224 cho tập {split.upper()} ({len(df)} ảnh)...")
    for idx, row in df.iterrows():
        src_path = row['image_path']
        # Tạo tên file hash duy nhất dựa trên index và filename
        cached_filename = f"{split}_{idx:04d}_{Path(src_path).name}"
        cached_path = cache_dir / cached_filename
        if not cached_path.exists():
            try:
                with Image.open(src_path) as img:
                    img_rgb = img.convert('RGB')
                    img_resized = img_rgb.resize((224, 224), Image.BILINEAR)
                    img_resized.save(cached_path, "JPEG", quality=95)
            except Exception as e:
                print(f"Lỗi ảnh {src_path}: {e}")
        total_images += 1

print(f"Hoàn thành cache 224x224 cho {total_images} ảnh trong {time.time()-t0:.2f}s!")
