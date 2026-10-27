# -*- coding: utf-8 -*-
"""
Script Phân tích Độ nhạy Trọng số (Sensitivity Analysis) của Điểm Hỗ trợ Quyết định Lai (Hybrid Support Score)
Thực hiện ĐỘC LẬP trên tập VALIDATION (113 ảnh - 12 cá thể) - TUYỆT ĐỐI KHÔNG DÙNG TẬP TEST
Sinh viên: Trương Thị Kim Yên - ĐH Nha Trang
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from ontology_reasoning import FishDiseaseReasoner

torch.set_num_threads(8)
device = torch.device("cpu")

# 1. Nạp mô hình chốt EfficientNet-B0 và nhãn
with open("models/class_to_idx.json", "r", encoding="utf-8") as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_to_idx))
model.load_state_dict(torch.load("models/efficientnet_b0_best.pth", map_location=device))
model.eval()

# 2. Khởi tạo bộ suy luận Ontology Reasoner
reasoner = FishDiseaseReasoner()

# 3. Chuẩn bị tập Validation
val_df = pd.read_csv("data/splits/val.csv")
cache_dir = Path("data/cache_224")

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Bảng triệu chứng thực tế tương ứng với nhãn ground truth lâm sàng
TYPICAL_SYMPTOMS_BY_CLASS = {
    "Bacterial Gill Diseases": ["SYM_FRAYED_GILLS", "SYM_LETHARGY"],
    "Bacterial Red Diseases": ["SYM_SKIN_ULCERS", "SYM_ABDOMINAL_DISTENSION"],
    "Epizootic Ulcerative Syndrome (EUS)": ["SYM_SKIN_ULCERS", "SYM_LETHARGY"],
    "Healthy Fish": ["SYM_NORMAL_BEHAVIOR"]
}

# Đo đạc môi trường thực nghiệm tương ứng với điều kiện bệnh học
TYPICAL_WATER_BY_CLASS = {
    "Bacterial Gill Diseases": {"temperature": 29.0, "ph": 7.4, "dissolved_oxygen": 2.9, "ammonia": 0.6},
    "Bacterial Red Diseases": {"temperature": 30.5, "ph": 7.6, "dissolved_oxygen": 4.0, "ammonia": 0.7},
    "Epizootic Ulcerative Syndrome (EUS)": {"temperature": 22.0, "ph": 5.9, "dissolved_oxygen": 4.2, "ammonia": 0.2},
    "Healthy Fish": {"temperature": 28.0, "ph": 7.5, "dissolved_oxygen": 5.2, "ammonia": 0.1}
}

# 4. Dự đoán xác suất AI cho toàn bộ 113 ảnh Validation
val_predictions = []
with torch.no_grad():
    for idx, row in val_df.iterrows():
        cached_p = cache_dir / f"val_{idx:04d}_{Path(row['image_path']).name}"
        img = Image.open(cached_p).convert('RGB')
        t_img = val_transform(img).unsqueeze(0).to(device)
        out = model(t_img)
        probs = F.softmax(out, dim=1).numpy()[0]
        
        # Danh sách Top-k dự đoán
        top_k = []
        for i in range(len(probs)):
            top_k.append({"class_name": idx_to_class[i], "confidence": float(probs[i])})
        top_k.sort(key=lambda x: x["confidence"], reverse=True)
        
        val_predictions.append({
            "idx": idx,
            "filename": Path(row['image_path']).name,
            "true_class": row['class_name'],
            "specimen_id": row['specimen_id'],
            "top_predictions": top_k,
            "top1_class": top_k[0]["class_name"],
            "top1_conf": top_k[0]["confidence"]
        })

print(f"Đã hoàn thành dự đoán cho toàn bộ {len(val_predictions)} ảnh tập Validation!")

# 5. Định nghĩa các cấu hình trọng số cần thử nghiệm (Sensitivity Configurations)
weight_configs = [
    {
        "config_name": "Config_1_VisionHeavy",
        "label": "Ưu tiên Thị giác (0.60 - 0.30 - 0.10)",
        "w_ai": 0.60, "w_symptom": 0.30, "w_water": 0.10,
        "rationale": "Ưu tiên mô hình thị giác AI; triệu chứng và môi trường đóng vai trò kiểm tra phụ trợ."
    },
    {
        "config_name": "Config_2_BalancedBaseline",
        "label": "Cân bằng Lâm sàng (0.50 - 0.35 - 0.15)",
        "w_ai": 0.50, "w_symptom": 0.35, "w_water": 0.15,
        "rationale": "Cấu hình cân bằng lâm sàng: AI giữ vai trò định hướng chính (0.50), triệu chứng xác nhận thứ cấp (0.35), môi trường đóng vai trò xúc tác/kích phát (0.15)."
    },
    {
        "config_name": "Config_3_ClinicalHeavy",
        "label": "Ưu tiên Lâm sàng & Dịch tễ (0.40 - 0.40 - 0.20)",
        "w_ai": 0.40, "w_symptom": 0.40, "w_water": 0.20,
        "rationale": "Đồng đều AI và triệu chứng lâm sàng (mỗi bên 0.40), tăng tỷ trọng môi trường nước lên 0.20."
    },
    {
        "config_name": "Config_4_EqualWeights",
        "label": "Trọng số Cân bằng Đều (0.33 - 0.33 - 0.34)",
        "w_ai": 0.333, "w_symptom": 0.333, "w_water": 0.334,
        "rationale": "Coi 3 nguồn bằng chứng có giá trị đóng góp ngang nhau trong quyết định chẩn đoán."
    }
]

# 6. Chạy thử nghiệm và phân tích độ nhạy
results_by_config = []
sample_scores_matrix = []

for cfg in weight_configs:
    cfg_name = cfg["config_name"]
    w_dict = {"w_ai": cfg["w_ai"], "w_symptom": cfg["w_symptom"], "w_water": cfg["w_water"]}
    
    scores = []
    correct_inferred = 0
    conflict_count = 0
    
    for sample in val_predictions:
        t_cls = sample["true_class"]
        syms = TYPICAL_SYMPTOMS_BY_CLASS.get(t_cls, ["SYM_LETHARGY"])
        water = TYPICAL_WATER_BY_CLASS.get(t_cls, {})
        
        res = reasoner.reason(
            ai_predictions=sample["top_predictions"],
            observed_symptoms=syms,
            water_measurements=water,
            weight_config=w_dict
        )
        
        inferred = res["case_summary"]["inferred_disease"]
        s_score = res["case_summary"]["support_score"]
        has_conf = res["case_summary"]["conflict_detected"]
        
        scores.append(s_score)
        if inferred.lower() == t_cls.lower():
            correct_inferred += 1
        if has_conf:
            conflict_count += 1
            
        sample_scores_matrix.append({
            "sample_idx": sample["idx"],
            "true_class": t_cls,
            "config_name": cfg_name,
            "config_label": cfg["label"],
            "support_score": s_score,
            "is_correct_inferred": inferred.lower() == t_cls.lower(),
            "conflict_detected": has_conf
        })
        
    scores = np.array(scores)
    accuracy = correct_inferred / len(val_predictions)
    
    results_by_config.append({
        "Config Name": cfg_name,
        "Tên cấu hình": cfg["label"],
        "w_AI": cfg["w_ai"],
        "w_Symptom": cfg["w_symptom"],
        "w_Water": cfg["w_water"],
        "Concordance Rate (Độ khớp GT)": round(accuracy, 4),
        "Mean Support Score": round(float(np.mean(scores)), 4),
        "Std Support Score": round(float(np.std(scores)), 4),
        "Min Score": round(float(np.min(scores)), 4),
        "Max Score": round(float(np.max(scores)), 4),
        "Median Score": round(float(np.median(scores)), 4),
        "Số ca kích hoạt mâu thuẫn": conflict_count,
        "Cơ sở lý giải": cfg["rationale"]
    })

# 7. Xuất file báo cáo Sensitivity Analysis
reports_dir = Path("reports")
reports_dir.mkdir(parents=True, exist_ok=True)

df_summary = pd.DataFrame(results_by_config)
out_csv = reports_dir / "hybrid_score_sensitivity_analysis.csv"
df_summary.to_csv(out_csv, index=False)
print(f"Đã lưu kết quả phân tích độ nhạy tại: {out_csv}")
print("\nBẢNG KẾT QUẢ PHÂN TÍCH ĐỘ NHẠY TRỌNG SỐ TRÊN TẬP VALIDATION:")
print(df_summary[["Tên cấu hình", "Concordance Rate (Độ khớp GT)", "Mean Support Score", "Std Support Score", "Min Score", "Max Score"]].to_string(index=False))

# 8. Trực quan hóa phân bố điểm số qua các cấu hình
df_samples = pd.DataFrame(sample_scores_matrix)

plt.figure(figsize=(10, 6), dpi=300)
palette = sns.color_palette("Set2", len(weight_configs))

sns.boxplot(data=df_samples, x="config_label", y="support_score", hue="config_label", palette=palette, width=0.45, legend=False)
sns.stripplot(data=df_samples, x="config_label", y="support_score", color="black", alpha=0.3, jitter=0.2, size=4)

plt.title("Phân tích Độ nhạy Điểm Hỗ trợ Quyết định (Rule-based Support Score)\ntrên tập Validation (113 ảnh - 4 Cấu hình Trọng số)", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Cấu hình Trọng số Thử nghiệm", fontsize=11, labelpad=8)
plt.ylabel("Điểm Hỗ trợ Quyết định (Hybrid Support Score)", fontsize=11, labelpad=8)
plt.xticks(rotation=12, ha='right', fontsize=9.5)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

out_plot = reports_dir / "hybrid_score_sensitivity_plot.png"
plt.savefig(out_plot)
plt.close()
print(f"Đã lưu biểu đồ phân tích độ nhạy tại: {out_plot}")
