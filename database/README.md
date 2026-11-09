# 🗄️ Cơ Sở Dữ Liệu Chẩn Đoán Bệnh Cá (FishDiagnosis Database)

Thư mục này quản lý cấu trúc bảng, lược đồ DDL và script khởi tạo dữ liệu cho hệ thống lưu trữ ca bệnh và kiểm toán (Audit Trail).

---

## 1. Danh mục Tệp tin

| Tệp tin | Định dạng | Mô tả chức năng |
| :--- | :---: | :--- |
| **`schema.sql`** | SQL (DDL) | Kịch bản tạo toàn bộ 10 bảng quan hệ, ràng buộc khóa ngoại `FOREIGN KEY` và chỉ mục tìm kiếm tối ưu. |
| **`init_db.py`** | Python Script | Script tự động khởi tạo CSDL SQLite và nạp dữ liệu nền tảng (Users, 5 Symptoms chuẩn, Model Run chính thức). |
| **`FishDiagnosis.db`** | SQLite 3 | Tệp cơ sở dữ liệu đã khởi tạo sẵn kèm các ca mẫu phục vụ chạy thử nghiệm ngay. |
| **`FishDiagnosis.sqbpro`** | DB Browser Project | Tệp dự án mở bằng phần mềm DB Browser for SQLite để xem trực quan cấu trúc và dữ liệu. |

---

## 2. Hướng dẫn Khởi tạo Lại CSDL từ Đầu (Database Reproduction)

Bất kỳ ai sao chép mã nguồn từ GitHub đều có thể tái tạo lại cơ sở dữ liệu hoàn chỉnh từ đầu thông qua một câu lệnh duy nhất mà không bị phụ thuộc vào tệp `.db` có sẵn:

```bash
# Khởi tạo CSDL mới (tự động áp dụng schema.sql và nạp master seed data)
python database/init_db.py

# Hoặc nếu muốn xóa sạch CSDL cũ và tạo mới hoàn toàn từ đầu:
python database/init_db.py --force
```

---

## 3. Cấu trúc 10 Bảng Quan hệ trong Cơ sở Dữ liệu

1. **`users`:** Quản lý tài khoản người dùng và vai trò nghiệp vụ (Kỹ thuật viên thủy sản, Chuyên gia bệnh học).
2. **`diagnostic_cases`:** Hồ sơ phiên chẩn đoán (`case_id`, ngày giờ `created_at`, mã loài cá `fish_species_uri`, ghi chú ngữ cảnh ao nuôi `note`).
3. **`case_images`:** Quản lý tệp ảnh cá nghi bệnh tải lên hệ thống (`file_path`, `original_name`).
4. **`model_runs`:** Lưu vết thông tin và phiên bản mô hình Deep Learning (`EfficientNet-B0 v1.0`, các chỉ số thực nghiệm Test Accuracy/Macro-F1).
5. **`ai_predictions`:** Lưu nhãn dự đoán của AI (`predicted_label`), điểm số độ tin cậy (`confidence_score`) và vector phân phối xác suất Top-3 (`topk_json`).
6. **`symptoms`:** Danh mục triệu chứng bệnh lý chuẩn hóa kèm mã URI ánh xạ sang Ontology (`symptom_code`, `symptom_name`, `symptom_uri`).
7. **`case_symptoms`:** Lưu các triệu chứng lâm sàng được ghi nhận thực tế trên từng ca chẩn đoán.
8. **`water_measurements`:** Lưu trữ chi tiết từng thông số chất lượng môi trường nước ao nuôi theo trường riêng biệt (`temperature`, `ph`, `dissolved_oxygen`, `ammonia`, `nitrite`, `alkalinity`, `turbidity`).
9. **`reasoning_results`:** Kết quả suy luận từ Ontology và hệ thống lai (`disease_uri`, Điểm Hỗ trợ Hỗn hợp `support_score`, bằng chứng ủng hộ `evidence_json`, cờ và dạng mâu thuẫn `conflict_json`, phiên bản Ontology `ontology_version`).
10. **`recommendations`:** Lưu trữ toàn văn phác đồ điều trị kháng sinh/thuốc tắm và khuyến nghị xử lý nước cấp thời chuẩn theo khuyến cáo của FAO và WOAH.
