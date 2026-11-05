# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')
from docx import Document

DOC_PATH = "d:/Project/DoAnDraft_v1.docx"
doc = Document(DOC_PATH)

updated_p233 = False
updated_p236 = False

for i, p in enumerate(doc.paragraphs):
    # Cập nhật đoạn tóm tắt Test metrics (Para 236)
    if "77,75%" in p.text or "77.75%" in p.text or "Test Macro-F1: 77" in p.text:
        print(f"Found old test metrics in Para {i}: {p.text}")
        p.text = "Test Accuracy: 82,67% | Test Macro-F1: 85,12% | Test Macro-Precision: 87,43% | Test Macro-Recall: 85,27% | Test Weighted-F1: 82,06%"
        print(f" -> Updated to: {p.text}")
        updated_p236 = True
        
    # Cập nhật đoạn giải trình chọn checkpoint (Para 233)
    if "4.3.3. Lý do chọn EfficientNet-B0" in doc.paragraphs[i-1].text:
        print(f"Found Section 4.3.3 intro in Para {i}: {p.text[:80]}...")
        new_text = (
            "Checkpoint chính thức được xác định cho toàn bộ hệ thống là models/efficientnet_b0_best.pth "
            "(kích thước 16,35 MB, 4,01 triệu tham số). Tuân thủ phương pháp luận khoa học, checkpoint này được chốt "
            "hoàn toàn dựa trên kết quả thực nghiệm trên tập Validation (113 ảnh – 13 cá thể) qua 5 random seeds "
            "(42, 100, 2024, 777, 999), đạt Val Macro-F1 trung bình 0.7648 ± 0.0303. Mặc dù ResNet-50 có Val Macro-F1 "
            "trung bình cao hơn đôi chút (0.7823), nhưng mô hình này có độ biến động lớn hơn nhiều giữa các hạt giống "
            "(độ lệch chuẩn ±0.0452 so với ±0.0303 của EfficientNet-B0), thể hiện tính kém ổn định hơn. Đồng thời, "
            "ResNet-50 sở hữu tới 23,52 triệu tham số và thời gian suy luận lên tới 112,99 ms/ảnh trên CPU, gấp gần 6 lần "
            "về kích thước và gần 2 lần về độ trễ so với EfficientNet-B0 (4,01M tham số, độ trễ 61,3 ms/ảnh, tương đương 16,3 FPS). "
            "Do đó, EfficientNet-B0 là sự lựa chọn tối ưu nhất về tương quan giữa độ chính xác, độ ổn định và chi phí tính toán "
            "để triển khai thực tế trên máy tính trang trại hoặc thiết bị không có GPU chuyên dụng."
        )
        p.text = new_text
        print(f" -> Updated Para {i} with detailed checkpoint selection explanation.")
        updated_p233 = True

if updated_p236 and updated_p233:
    doc.save(DOC_PATH)
    print(f"\n[THÀNH CÔNG] Đã lưu file cập nhật: {DOC_PATH}")
else:
    print(f"\n[CẢNH BÁO] Chưa cập nhật đầy đủ (p233={updated_p233}, p236={updated_p236})")
