# 📦 Quản lý Trọng số Mô hình (Model Checkpoints)

Thư mục này chứa cấu hình nhãn phân loại và tệp trọng số (checkpoint) chính thức của mô hình học sâu **EfficientNet-B0** được lựa chọn cho hệ thống.

---

## 1. Tệp Trọng số Chính thức (Official Checkpoint)

| Tệp tin | Định dạng | Dung lượng | Mô tả |
| :--- | :---: | :---: | :--- |
| **`class_to_idx.json`** | JSON | 143 B | Ánh xạ 4 lớp phân loại: `{BGD: 0, BRD: 1, EUS: 2, Healthy: 3}` |
| **`efficientnet_b0_best.zip`** | ZIP Archive | **14.39 MB** | Checkpoint trọng số đã nén (được lưu trực tiếp trên GitHub) |
| **`efficientnet_b0_best.pth`** | PyTorch Weights | **16.35 MB** | Tệp trọng số gốc (được tự động giải nén khi khởi chạy hệ thống) |

---

## 2. Cơ chế Tự động Giải nén (Auto-Extract Mechanism)

Khi bạn sao chép (clone) kho mã nguồn từ GitHub về máy tính:
- Kho chứa sẵn tệp nén **`models/efficientnet_b0_best.zip`** (14.39 MB, dưới ngưỡng giới hạn 100MB của GitHub).
- Khi bạn khởi chạy ứng dụng Streamlit (`streamlit run app/main.py`) hoặc chạy kiểm thử trong Python (`from src.predictor import FishDiseasePredictor`), module `FishDiseasePredictor` sẽ **tự động phát hiện và giải nén** tệp `efficientnet_b0_best.pth` trong khoảng **0.2 giây** nếu tệp `.pth` chưa tồn tại.
- Bạn **không cần** phải thực hiện bất kỳ thao tác tải hay giải nén thủ công nào!

### Thao tác giải nén thủ công (Tùy chọn):
Nếu muốn giải nén thủ công trước khi chạy:
```bash
# Trên Windows PowerShell
Expand-Archive -Path models/efficientnet_b0_best.zip -DestinationPath models/ -Force

# Trên Linux / macOS
unzip -o models/efficientnet_b0_best.zip -d models/
```

---

## 3. Thông số Hiệu năng của Checkpoint Chính thức

- **Kiến trúc mô hình:** EfficientNet-B0 (Compound Scaling Baseline)
- **Số lượng tham số:** 4,01 triệu tham số (4.01M params)
- **Độ trễ suy luận (Inference Latency):** ~61.3 ms/ảnh trên CPU Intel Core i7 (Tốc độ: 16.3 FPS)
- **Cơ sở lựa chọn:** Đánh giá độ ổn định trên tập **Validation (113 ảnh - 13 cá thể)** qua **5 random seeds**: $\text{Val Macro-F1} = 0.7648 \pm 0.0303$.
- **Kết quả đánh giá trên tập Test độc lập (Held-out Test: 225 ảnh - 16 cá thể):**
  - **Accuracy:** `82.67%` (186/225 ảnh)
  - **Macro-Precision:** `87.43%`
  - **Macro-Recall:** `85.27%`
  - **Macro-F1:** `85.12%`
  - **Weighted-F1:** `82.06%`
  - **Bacterial Gill Diseases (BGD):** F1 = `98.36%` (Support = 31)
  - **Bacterial Red Diseases (BRD):** F1 = `72.86%` (Support = 83)
  - **Epizootic Ulcerative Syndrome (EUS):** F1 = `85.29%` (Support = 35)
  - **Healthy Fish (Cá khỏe):** F1 = `83.98%` (Support = 76)
