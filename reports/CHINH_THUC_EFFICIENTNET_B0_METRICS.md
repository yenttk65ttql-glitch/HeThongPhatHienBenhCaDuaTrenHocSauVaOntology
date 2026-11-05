# BÁO CÁO XÁC LẬP BỘ CHỈ SỐ TEST CHÍNH THỨC CỦA MÔ HÌNH EFFICIENTNET-B0

- **Đề tài:** Hệ thống hỗ trợ phát hiện bệnh cá dựa trên học sâu và ontology
- **Học viên/Sinh viên:** Trương Thị Kim Yên (MSSV: 65134516) - ĐH Nha Trang
- **Tệp checkpoint chính thức duy nhất:** `models/efficientnet_b0_best.pth`
- **Kích thước checkpoint:** 16.35 MB (4.01M tham số)

---

## 1. Cơ sở lựa chọn Checkpoint trên tập Validation (Selection Protocol)
Tuân thủ nguyên tắc phương pháp luận khoa học: **Tập Test được khóa hoàn toàn (vaulted), không dùng để tinh chỉnh hay lựa chọn mô hình**.

Mô hình và checkpoint được lựa chọn hoàn toàn dựa trên tập **Validation (113 ảnh - 13 cá thể)** qua **5 random seeds** (42, 100, 2024, 777, 999):
- **Val Macro-F1 trung bình:** `0.7648 ± 0.0303` (độ lệch chuẩn thấp, tính ổn định cao).
- **So sánh với các kiến trúc đối chuẩn:**
  + **ResNet-50:** Val Macro-F1 = 0.7823 ± 0.0452 (biến động hạt giống lớn hơn, 23.52M tham số, độ trễ 112.99 ms).
  + **ResNet-18:** Val Macro-F1 = 0.7559 ± 0.0182 (11.18M tham số, độ trễ 51.18 ms).
  + **MobileNetV3:** Val Macro-F1 = 0.6087 ± 0.0300 (1.52M tham số, độ trễ 26.54 ms - hiệu năng phân loại thấp trên tập dữ liệu này).
- **Kết luận lựa chọn:** `EfficientNet-B0` đạt điểm cân bằng tối ưu giữa độ chính xác, độ ổn định thực nghiệm, kích thước nhỏ gọn (4.01M tham số) và tốc độ suy luận nhanh (~61 ms/ảnh trên CPU), đặc biệt phù hợp triển khai thực tế trên máy tính trang trại không có GPU.

---

## 2. Kết quả Đánh giá Chính thức Duy nhất trên Tập Test (Held-out Test: 225 ảnh - 16 cá thể)

### 2.1. Chỉ số tổng thể:
- **Test Accuracy:** `82.67%` (0.8267)
- **Test Macro-Precision:** `87.43%` (0.8743)
- **Test Macro-Recall:** `85.27%` (0.8527)
- **Test Macro-F1:** `85.12%` (0.8512)
- **Test Weighted-F1:** `82.06%` (0.8206)

### 2.2. Chi tiết từng lớp bệnh (Per-Class Performance):
| Lớp bệnh | Precision | Recall | F1-Score | Số mẫu Test (Support) |
| :--- | :---: | :---: | :---: | :---: |
| **Bacterial Gill Diseases (BGD)** | 100.00% | 96.77% | 98.36% | 31 |
| **Bacterial Red Diseases (BRD)** | 89.47% | 61.45% | 72.86% | 83 |
| **Epizootic Ulcerative Syndrome (EUS)** | 87.88% | 82.86% | 85.29% | 35 |
| **Healthy Fish (Cá khỏe)** | 72.38% | 100.00% | 83.98% | 76 |
| **Macro Average** | **87.43%** | **85.27%** | **85.12%** | **225** |
| **Weighted Average** | **84.90%** | **82.67%** | **82.06%** | **225** |

### 2.3. Ma trận nhầm lẫn (Confusion Matrix):
```
True \ Pred   BGD   BRD   EUS   Healthy
BGD             30     0     0     1
BRD              0    51     4    28
EUS              0     6    29     0
Healthy          0     0     0    76
```

---

## 3. Rà soát và Giải trình Sai lệch Số liệu cũ (Con số 77,75%)
1. **Nguồn gốc số 77,75%:** Đây là kết quả Test Macro-F1 thu được từ đợt chạy sơ bộ ngày 15/10/2026 (trước khi hoàn thiện tiền xử lý lọc trùng lặp pHash và chia dữ liệu phân tầng theo cá thể).
2. **Khắc phục:** 
   - Đã di chuyển toàn bộ các tệp kết quả sơ bộ ngày 15/10 vào thư mục `reports/archive/`.
   - Cập nhật và đồng nhất giá trị trong văn bản đồ án `DoAnDraft_v1.docx`: Thống nhất chính xác con số **Test Macro-F1 = 85,12%** và **Test Accuracy = 82,67%**.
