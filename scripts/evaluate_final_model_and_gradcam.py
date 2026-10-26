# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import confusion_matrix, classification_report

torch.set_num_threads(8)
device = torch.device("cpu")

# 1. Đọc ánh xạ nhãn
with open("models/class_to_idx.json", "r", encoding="utf-8") as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}
class_names = [idx_to_class[i] for i in range(len(class_to_idx))]

# 2. Định nghĩa Test Dataset với Cached 224x224
class TestCachedDataset(Dataset):
    def __init__(self, transform=None):
        self.df = pd.read_csv("data/splits/test.csv")
        self.transform = transform
        self.cache_dir = Path("data/cache_224")
        self.cached_files = []
        for idx, row in self.df.iterrows():
            cached_filename = f"test_{idx:04d}_{Path(row['image_path']).name}"
            self.cached_files.append(self.cache_dir / cached_filename)
            
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.cached_files[idx]
        image = Image.open(img_path).convert('RGB')
        label = class_to_idx[row['class_name']]
        
        if self.transform:
            tensor_img = self.transform(image)
        else:
            tensor_img = transforms.ToTensor()(image)
            
        return tensor_img, label, str(img_path), row['image_path'], row['class_name']

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_ds = TestCachedDataset(transform=val_transform)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# 3. Tải mô hình chốt EfficientNet-B0
print(">>> Khởi tạo và nạp trọng số mô hình chốt: EFFICIENTNET_B0...")
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_to_idx))
ckpt_path = "models/efficientnet_b0_best.pth"
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()

# 4. Dự đoán toàn bộ tập Test và tính Top-3 probabilities
all_targets = []
all_preds = []
all_probs = []
sample_info = []

with torch.no_grad():
    for imgs, labels, cached_paths, orig_paths, true_names in test_loader:
        outputs = model(imgs)
        probs = F.softmax(outputs, dim=1).numpy()
        preds = outputs.argmax(1).numpy()
        
        all_targets.extend(labels.numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)
        
        for i in range(len(labels)):
            sample_info.append({
                "orig_path": orig_paths[i],
                "cached_path": cached_paths[i],
                "true_idx": int(labels[i]),
                "true_class": true_names[i],
                "pred_idx": int(preds[i]),
                "pred_class": idx_to_class[preds[i]],
                "probs": probs[i]
            })

all_targets = np.array(all_targets)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# 5. Vẽ và xuất CONFUSION MATRIX
reports_dir = Path("reports")
reports_dir.mkdir(parents=True, exist_ok=True)

cm = confusion_matrix(all_targets, all_preds)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Rút ngắn tên hiển thị trên ma trận nhầm lẫn cho đẹp mắt
display_names = [
    "BGD\n(Mang)",
    "BRD\n(Đốm đỏ)",
    "EUS\n(Lở loét)",
    "Healthy\n(Khỏe mạnh)"
]

plt.figure(figsize=(8, 6.5), dpi=300)
annot = np.empty_like(cm).astype(str)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=True,
            xticklabels=display_names, yticklabels=display_names,
            linewidths=1.2, linecolor='white')
plt.title("Ma trận nhầm lẫn (Confusion Matrix) - EfficientNet-B0 (Tập Test: 225 ảnh)", fontsize=13, pad=14, fontweight='bold')
plt.xlabel("Nhãn dự đoán của mô hình (Predicted Label)", fontsize=11, labelpad=8)
plt.ylabel("Nhãn thực tế chuyên gia (True Label)", fontsize=11, labelpad=8)
plt.tight_layout()
cm_plot_path = reports_dir / "final_model_confusion_matrix.png"
plt.savefig(cm_plot_path)
plt.close()
print(f"Đã lưu Confusion Matrix tại: {cm_plot_path}")

# 6. Báo cáo Precision / Recall / F1 từng lớp
report_dict = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True, digits=4)
df_report = pd.DataFrame(report_dict).transpose()
report_csv_path = reports_dir / "final_model_per_class_metrics.csv"
df_report.to_csv(report_csv_path)
print(f"Đã lưu bảng Precision/Recall/F1 từng lớp tại: {report_csv_path}")
print("\nBÁO CÁO PHÂN LOẠI CHI TIẾT TỪNG LỚP TRÊN TẬP TEST:")
print(df_report)

# 7. Phân tích Top-3 Probabilities & Bảng ảnh phân loại sai điển hình
top3_records = []
misclassified_records = []

for s in sample_info:
    probs = s["probs"]
    # Sort descending
    top3_indices = np.argsort(probs)[::-1][:3]
    top3_classes = [idx_to_class[idx] for idx in top3_indices]
    top3_confidences = [probs[idx] for idx in top3_indices]
    
    rec = {
        "filename": Path(s["orig_path"]).name,
        "image_path": s["orig_path"],
        "cached_path": s["cached_path"],
        "true_class": s["true_class"],
        "predicted_class": s["pred_class"],
        "is_correct": s["true_idx"] == s["pred_idx"],
        "top1_class": top3_classes[0],
        "top1_confidence": round(float(top3_confidences[0]), 4),
        "top2_class": top3_classes[1],
        "top2_confidence": round(float(top3_confidences[1]), 4),
        "top3_class": top3_classes[2],
        "top3_confidence": round(float(top3_confidences[2]), 4),
        "margin_top1_top2": round(float(top3_confidences[0] - top3_confidences[1]), 4)
    }
    top3_records.append(rec)
    if not rec["is_correct"]:
        misclassified_records.append(rec)

df_top3 = pd.DataFrame(top3_records)
df_top3.to_csv(reports_dir / "test_predictions_top3.csv", index=False)

df_mis = pd.DataFrame(misclassified_records)
df_mis.to_csv(reports_dir / "misclassified_samples_analysis.csv", index=False)
print(f"\nTổng số mẫu tập Test: {len(df_top3)} | Số mẫu phân loại sai: {len(df_mis)} ({len(df_mis)/len(df_top3)*100:.2f}%)")
print(f"Đã lưu danh sách phân loại sai tại: {reports_dir / 'misclassified_samples_analysis.csv'}")

# 8. Cài đặt Grad-CAM cho EfficientNet-B0
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Đăng ký hook
        self.hook_fwd = target_layer.register_forward_hook(self._forward_hook)
        self.hook_bwd = target_layer.register_full_backward_hook(self._backward_hook)
        
    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
        
    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()
        
    def generate(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        score = output[0, target_class]
        score.backward()
        
        # Global Average Pooling trên gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True) # [1, C, 1, 1]
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True) # [1, 1, H, W]
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        
        # Chuẩn hóa về [0, 1]
        cam = cv2.resize(cam, (224, 224))
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam, target_class, F.softmax(output, dim=1).detach().cpu().numpy()[0]
        
    def remove_hooks(self):
        self.hook_fwd.remove()
        self.hook_bwd.remove()

# Lớp convolutional cuối cùng của EfficientNet-B0 là model.features[-1]
target_layer = model.features[-1]
grad_cam = GradCAM(model, target_layer)

# 9. Chọn các mẫu tiêu biểu: 4 mẫu ĐÚNG (mỗi lớp 1 mẫu) và 4 mẫu SAI điển hình
correct_samples = []
for c_name in class_names:
    matches = [s for s in top3_records if s["is_correct"] and s["true_class"] == c_name]
    if matches:
        # Chọn mẫu có độ tự tin cao nhất
        best_match = max(matches, key=lambda x: x["top1_confidence"])
        correct_samples.append(best_match)

# Lấy 4 mẫu sai tiêu biểu từ các cặp nhầm lẫn thường gặp:
# Ví dụ: BRD -> EUS, BGD -> Healthy, EUS -> BRD, Healthy -> BRD/BGD
mis_samples = []
target_pairs = [
    ("Bacterial Red Diseases", "Epizootic Ulcerative Syndrome (EUS)"),
    ("Epizootic Ulcerative Syndrome (EUS)", "Bacterial Red Diseases"),
    ("Bacterial Gill Diseases", "Healthy Fish"),
    ("Healthy Fish", "Bacterial Gill Diseases")
]

for true_c, pred_c in target_pairs:
    pair_matches = [s for s in misclassified_records if s["true_class"] == true_c and s["predicted_class"] == pred_c]
    if pair_matches:
        mis_samples.append(pair_matches[0])
        
# Nếu thiếu cặp nào thì lấy thêm từ misclassified_records
for s in misclassified_records:
    if len(mis_samples) >= 4:
        break
    if s not in mis_samples:
        mis_samples.append(s)

print(f"Đã chọn {len(correct_samples)} mẫu đúng và {len(mis_samples)} mẫu sai để trực quan hóa Grad-CAM.")

# Hàm vẽ lưới Grad-CAM
def plot_gradcam_grid(samples_list, title, save_filename, is_error_grid=False):
    n_samples = len(samples_list)
    fig, axes = plt.subplots(n_samples, 3, figsize=(11, 3.2 * n_samples), dpi=250)
    if n_samples == 1:
        axes = np.expand_dims(axes, 0)
        
    for i, s in enumerate(samples_list):
        img_p = s["cached_path"]
        orig_pil = Image.open(img_p).convert('RGB')
        orig_np = np.array(orig_pil)
        
        # Tensor hóa cho mô hình
        input_tensor = val_transform(orig_pil).unsqueeze(0).to(device)
        cam, pred_cls_idx, probs = grad_cam.generate(input_tensor, target_class=class_to_idx[s["predicted_class"]])
        
        # Tạo Heatmap màu bằng OpenCV
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Phủ lên ảnh gốc (Superimposed)
        superimposed = np.float32(heatmap) * 0.4 + np.float32(orig_np) * 0.6
        superimposed = np.uint8(superimposed)
        
        # Cột 1: Ảnh gốc
        axes[i, 0].imshow(orig_np)
        axes[i, 0].axis('off')
        if not is_error_grid:
            axes[i, 0].set_title(f"Ảnh gốc (True: {s['true_class']})\nFile: {s['filename']}", fontsize=9, fontweight='bold')
        else:
            axes[i, 0].set_title(f"Ảnh gốc (Thực tế: {s['true_class']})\nFile: {s['filename']}", fontsize=9, color='darkgreen', fontweight='bold')
            
        # Cột 2: Bản đồ nhiệt Grad-CAM
        axes[i, 1].imshow(cam, cmap='jet')
        axes[i, 1].axis('off')
        axes[i, 1].set_title(f"Bản đồ nhiệt Grad-CAM\n(Vùng mạng tập trung)", fontsize=9, fontweight='bold')
        
        # Cột 3: Ảnh phủ chồng
        axes[i, 2].imshow(superimposed)
        axes[i, 2].axis('off')
        if not is_error_grid:
            axes[i, 2].set_title(f"Dự đoán: {s['predicted_class']}\nĐộ tin cậy: {s['top1_confidence']*100:.1f}%", fontsize=9, color='navy', fontweight='bold')
        else:
            axes[i, 2].set_title(f"Dự đoán SAI: {s['predicted_class']} ({s['top1_confidence']*100:.1f}%)\nTop-2: {s['top2_class']} ({s['top2_confidence']*100:.1f}%)", fontsize=9, color='crimson', fontweight='bold')
            
    plt.suptitle(title, fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    out_p = reports_dir / save_filename
    plt.savefig(out_p, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu biểu đồ Grad-CAM tại: {out_p}")

# Xuất 2 biểu đồ Grad-CAM
plot_gradcam_grid(correct_samples, "Giải thích chẩn đoán Grad-CAM cho các ca phân loại ĐÚNG điển hình (EfficientNet-B0)", "gradcam_correct_samples.png", is_error_grid=False)
plot_gradcam_grid(mis_samples, "Phân tích nguyên nhân chẩn đoán Grad-CAM cho các ca phân loại SAI điển hình (EfficientNet-B0)", "gradcam_misclassified_samples.png", is_error_grid=True)

# Tạo ảnh tổng hợp 4 ca (2 đúng, 2 sai) cho báo cáo ngắn gọn
combined_samples = [correct_samples[1], correct_samples[2], mis_samples[0], mis_samples[1]]
plot_gradcam_grid(combined_samples, "Trực quan hóa vùng đặc trưng chẩn đoán Grad-CAM (Ca đúng & Ca sai)", "gradcam_analysis.png", is_error_grid=True)

grad_cam.remove_hooks()
print("\n>>> HOÀN THÀNH TẤT CẢ CÁC BƯỚC ĐÁNH GIÁ CHUYÊN SÂU CHO MÔ HÌNH CHỐN EFFICIENTNET-B0!")
