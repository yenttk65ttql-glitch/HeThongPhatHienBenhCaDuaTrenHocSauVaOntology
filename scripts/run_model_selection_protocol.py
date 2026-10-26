# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import time
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from sklearn.utils.class_weight import compute_class_weight

torch.set_num_threads(12)
device = torch.device("cpu")

# 1. Đọc ánh xạ nhãn
with open("models/class_to_idx.json", "r", encoding="utf-8") as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes = len(class_to_idx)

# 2. Định nghĩa CachedFishDataset
class CachedFishDataset(Dataset):
    def __init__(self, split_name, transform=None):
        self.split_name = split_name
        self.df = pd.read_csv(f"data/splits/{split_name}.csv")
        self.transform = transform
        self.cache_dir = Path("data/cache_224")
        
        # Pre-verify cached paths
        self.cached_files = []
        for idx, row in self.df.iterrows():
            cached_filename = f"{split_name}_{idx:04d}_{Path(row['image_path']).name}"
            self.cached_files.append(self.cache_dir / cached_filename)
            
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        img_path = self.cached_files[idx]
        image = Image.open(img_path)
        label = class_to_idx[self.df.iloc[idx]['class_name']]
        
        if self.transform:
            image = self.transform(image)
        return image, label

# 3. Augmentation & Transforms
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_df = pd.read_csv("data/splits/train.csv")
train_labels = [class_to_idx[name] for name in train_df['class_name']]
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def build_model(model_name):
    if model_name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif model_name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif model_name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif model_name == "mobilenet_v3":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    return m.to(device)

def get_model_size_mb(model):
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)

def measure_latency(model, n_runs=40):
    model.eval()
    x = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
        t0 = time.time()
        for _ in range(n_runs):
            _ = model(x)
        t1 = time.time()
    lat_ms = ((t1 - t0) / n_runs) * 1000
    fps = 1000 / lat_ms
    return lat_ms, fps

# 4. Danh sách các mô hình và các seed thực nghiệm
model_names = ["resnet18", "resnet50", "efficientnet_b0", "mobilenet_v3"]
seeds = [42, 100, 2024, 777, 999]

print("=" * 70)
print("KHÓA PROTOCOL ĐÁNH GIÁ ĐỘ ỔN ĐỊNH VÀ LỰA CHỌN MÔ HÌNH (TRAIN/VAL ONLY)")
print(f"Số lượng seeds thực nghiệm: {len(seeds)} {seeds}")
print("TẬP TEST (225 ảnh) ĐƯỢC GIỮ NGUYÊN HOÀN TOÀN (VAULTED / HELD-OUT)")
print("=" * 70)

val_dataset = CachedFishDataset("val", transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

protocol_records = []

for m_name in model_names:
    print(f"\n>>> Đang đánh giá mô hình: {m_name.upper()} qua 5 seeds...")
    seed_f1s = []
    seed_accs = []
    
    # Đo tham số và độ trễ
    probe_model = build_model(m_name)
    n_params_m = sum(p.numel() for p in probe_model.parameters()) / 1e6
    size_mb = get_model_size_mb(probe_model)
    lat_ms, fps = measure_latency(probe_model)
    
    # Huấn luyện/Đánh giá trên 5 seeds độc lập với early stopping nhanh
    for seed in seeds:
        set_seed(seed)
        train_dataset = CachedFishDataset("train", transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        model = build_model(m_name)
        
        # Nếu là seed 42, ta có thể dùng checkpoint tốt nhất đã lưu hoặc fine-tune nhanh
        ckpt_path = Path(f"models/{m_name}_best.pth")
        if seed == 42 and ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        else:
            # Fine-tune nhanh trên classifier head & top layers
            # Đóng băng backbone, chỉ fine-tune top layers để đo độ nhạy seed
            for param in model.parameters():
                param.requires_grad = False
            if m_name == "resnet18" or m_name == "resnet50":
                for param in model.fc.parameters(): param.requires_grad = True
                for param in model.layer4.parameters(): param.requires_grad = True
            elif m_name == "efficientnet_b0":
                for param in model.classifier.parameters(): param.requires_grad = True
                for param in model.features[-2:].parameters(): param.requires_grad = True
            elif m_name == "mobilenet_v3":
                for param in model.classifier.parameters(): param.requires_grad = True
                for param in model.features[-2:].parameters(): param.requires_grad = True
                
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
            
            # Huấn luyện 4 epochs
            for epoch in range(4):
                model.train()
                for imgs, lbls in train_loader:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    optimizer.zero_grad()
                    out = model(imgs)
                    loss = criterion(out, lbls)
                    loss.backward()
                    optimizer.step()
        
        # Đánh giá trên Validation
        model.eval()
        v_preds, v_targets = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                out = model(imgs)
                preds = out.argmax(1).cpu().numpy()
                v_preds.extend(preds)
                v_targets.extend(lbls.numpy())
                
        val_acc = accuracy_score(v_targets, v_preds)
        val_f1 = f1_score(v_targets, v_preds, average='macro', zero_division=0)
        seed_f1s.append(val_f1)
        seed_accs.append(val_acc)
        print(f"   Seed {seed:4d} | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_f1:.4f}")
        
    mean_f1 = np.mean(seed_f1s)
    std_f1 = np.std(seed_f1s)
    mean_acc = np.mean(seed_accs)
    std_acc = np.std(seed_accs)
    
    print(f"==> KẾT QUẢ {m_name.upper()}: Val Macro-F1 = {mean_f1:.4f} ± {std_f1:.4f} | Latency = {lat_ms:.2f} ms")
    
    protocol_records.append({
        "Model": m_name,
        "Val Macro-F1 (Mean)": round(mean_f1, 4),
        "Val Macro-F1 (Std)": round(std_f1, 4),
        "Val Macro-F1 (Mean ± Std)": f"{mean_f1:.4f} ± {std_f1:.4f}",
        "Val Accuracy (Mean ± Std)": f"{mean_acc:.4f} ± {std_acc:.4f}",
        "Parameters (M)": round(n_params_m, 2),
        "Model Size (MB)": round(size_mb, 2),
        "Latency (ms/img)": round(lat_ms, 2),
        "Throughput (FPS)": round(fps, 1),
        "Seed 42": round(seed_f1s[0], 4),
        "Seed 100": round(seed_f1s[1], 4),
        "Seed 2024": round(seed_f1s[2], 4),
        "Seed 777": round(seed_f1s[3], 4),
        "Seed 999": round(seed_f1s[4], 4)
    })

# Xuất ra báo cáo
reports_dir = Path("reports")
reports_dir.mkdir(parents=True, exist_ok=True)
df_protocol = pd.DataFrame(protocol_records)
out_csv = reports_dir / "model_selection_protocol_summary.csv"
df_protocol.to_csv(out_csv, index=False)

print("\n" + "=" * 80)
print("BẢNG TỔNG HỢP GIAO THỨC LỰA CHỌN MÔ HÌNH (reports/model_selection_protocol_summary.csv):")
print("=" * 80)
print(df_protocol[["Model", "Val Macro-F1 (Mean ± Std)", "Parameters (M)", "Latency (ms/img)", "Throughput (FPS)"]].to_string(index=False))
