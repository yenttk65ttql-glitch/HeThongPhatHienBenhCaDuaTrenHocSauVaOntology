# -*- coding: utf-8 -*-
"""
Script đồng bộ tệp báo cáo hoàn thiện DoAnDraft_v1_Updated.docx vào DoAnDraft_v1.docx
Sử dụng: Chạy sau khi người dùng đã đóng cửa sổ Word đang mở tệp DoAnDraft_v1.docx.
"""
import os
import sys
import shutil

sys.stdout.reconfigure(encoding='utf-8')

SRC = "DoAnDraft_v1_Updated.docx"
DST = "DoAnDraft_v1.docx"

if not os.path.exists(SRC):
    print(f"Lỗi: Không tìm thấy tệp nguồn {SRC}!")
    sys.exit(1)

try:
    shutil.copyfile(SRC, DST)
    print(f"Thành công: Đã đồng bộ toàn bộ nội dung hoàn thiện vào {DST}!")
except PermissionError:
    print(f"Thông báo: Tệp {DST} vẫn đang được mở bởi Microsoft Word hoặc một phần mềm khác.")
    print("Vui lòng đóng tệp Word và chạy lại lệnh này để hoàn tất đồng bộ.")
except Exception as e:
    print(f"Lỗi không xác định: {e}")
