# -*- coding: utf-8 -*-
"""
Script Khởi tạo và Thiết lập Cơ sở dữ liệu SQLite (FishDiagnosis.db)
Đề tài: Hệ thống Hỗ trợ Chẩn đoán Bệnh Cá Dựa trên Học Sâu và Ontology
Sinh viên: Trương Thị Kim Yên (MSSV: 65134516) - ĐH Nha Trang

Cách dùng:
    python database/init_db.py
    python database/init_db.py --force (Xóa tạo lại từ đầu)
"""

import os
import sys
import json
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime

# Đảm bảo in UTF-8 trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = PROJECT_ROOT / "database"
DB_PATH = DB_DIR / "FishDiagnosis.db"
SCHEMA_PATH = DB_DIR / "schema.sql"

def init_database(force: bool = False):
    DB_DIR.mkdir(parents=True, exist_ok=True)
    
    if force and DB_PATH.exists():
        print(f"[CẢNH BÁO] Đang xóa cơ sở dữ liệu cũ theo cờ --force: {DB_PATH}")
        DB_PATH.unlink()
        
    print(f">> Bắt đầu khởi tạo CSDL tại: {DB_PATH}")
    
    # 1. Đọc tệp schema.sql
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file schema tại: {SCHEMA_PATH}")
        
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema_sql = f.read()
        
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA foreign_keys = ON;")
    cursor = conn.cursor()
    
    # 2. Thực thi DDL
    cursor.executescript(schema_sql)
    print("✓ Đã thực thi thành công schema.sql (10 bảng và các chỉ mục).")
    
    # 3. Nạp dữ liệu nền tảng (Seed Master Data)
    # 3.1. Người dùng mặc định
    cursor.execute("SELECT count(*) FROM users;")
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
            INSERT INTO users (user_id, full_name, role, email) 
            VALUES (1, 'Kỹ thuật viên Trương Thị Kim Yên', 'Kỹ thuật viên Thủy sản', 'kimyen@ntu.edu.vn')
        """)
        print("✓ Đã nạp User mặc định (ID 1: Kỹ thuật viên Trương Thị Kim Yên).")

    # 3.2. Danh mục triệu chứng chuẩn hóa (Mapping với Ontology)
    standard_symptoms = [
        ("SYM_FRAYED_GILLS", "Mang xơ xác, hoại tử tơ mang, dính nhiều nhớt", "http://www.semanticweb.org/fish-disease-ontology#Frayed_Gills"),
        ("SYM_SKIN_ULCERS", "Loét da, xuất huyết đỏ ở thân và gốc vây", "http://www.semanticweb.org/fish-disease-ontology#Skin_Ulcers"),
        ("SYM_ABDOMINAL_DISTENSION", "Chướng bụng, hậu môn sưng đỏ lồi ra ngoài", "http://www.semanticweb.org/fish-disease-ontology#Abdominal_Distension"),
        ("SYM_LETHARGY", "Bơi lờ đờ, mất định hướng, nổi đầu tách đàn ở tầng mặt", "http://www.semanticweb.org/fish-disease-ontology#Lethargy_Anorexia"),
        ("SYM_NORMAL", "Ngoại hình bình thường, da vây nguyên vẹn, phản xạ nhanh", "http://www.semanticweb.org/fish-disease-ontology#Normal_Behavior_Appearance"),
    ]
    for code, name, uri in standard_symptoms:
        cursor.execute("""
            INSERT OR IGNORE INTO symptoms (symptom_code, symptom_name, symptom_uri)
            VALUES (?, ?, ?)
        """, (code, name, uri))
    print(f"✓ Đã nạp danh mục {len(standard_symptoms)} triệu chứng lâm sàng chuẩn.")

    # 3.3. Bản ghi mô hình chính thức (EfficientNet-B0 v1.0)
    cursor.execute("SELECT count(*) FROM model_runs WHERE model_name = 'EfficientNet-B0';")
    if cursor.fetchone()[0] == 0:
        metrics_dict = {
            "checkpoint": "models/efficientnet_b0_best.pth",
            "parameters_m": 4.01,
            "test_accuracy": 0.8267,
            "test_macro_f1": 0.8512,
            "test_macro_precision": 0.8743,
            "test_macro_recall": 0.8527,
            "validation_macro_f1_mean_std": "0.7648 ± 0.0303",
            "held_out_test_images": 225
        }
        cursor.execute("""
            INSERT INTO model_runs (model_name, version, metrics_json, created_at)
            VALUES (?, ?, ?, ?)
        """, ("EfficientNet-B0", "v1.0 (Official Best Checkpoint)", json.dumps(metrics_dict, ensure_ascii=False), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        print("✓ Đã nạp bản ghi mô hình chính thức: EfficientNet-B0 v1.0.")

    # 3.4. Bản ghi ca mẫu khởi đầu (nếu DB hoàn toàn trống ca bệnh)
    cursor.execute("SELECT count(*) FROM diagnostic_cases;")
    if cursor.fetchone()[0] == 0:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Ca mẫu 1: BGD đồng thuận
        cursor.execute("""
            INSERT INTO diagnostic_cases (user_id, created_at, fish_species_uri, note)
            VALUES (1, ?, 'Cá mè trắng (Hypophthalmichthys molitrix)', 'Ca mẫu kiểm thử ban đầu: Cá nổi đầu ở tầng mặt sáng sớm')
        """, (now_str,))
        case_id = cursor.lastrowid
        
        cursor.execute("""
            INSERT INTO case_images (case_id, file_path, original_name)
            VALUES (?, 'data/raw/Bacterial Gill Diseases/Bacterial Gill Diseases/Silver_carp_(Hypophthalmichthys molitrix)/Fish_19/IMG20260206071503.jpg', 'demo_bgd.jpg')
        """, (case_id,))
        img_id = cursor.lastrowid
        
        top3_json = json.dumps([
            {"class_name": "Bacterial Gill Diseases", "confidence": 0.9994},
            {"class_name": "Bacterial Red Diseases", "confidence": 0.0003},
            {"class_name": "Healthy Fish", "confidence": 0.0002}
        ], ensure_ascii=False)
        cursor.execute("""
            INSERT INTO ai_predictions (image_id, run_id, case_id, predicted_label, confidence_score, topk_json)
            VALUES (?, 1, ?, 'Bacterial Gill Diseases', 0.9994, ?)
        """, (img_id, case_id, top3_json))
        
        cursor.execute("INSERT INTO case_symptoms (case_id, symptom_code, selected_value) VALUES (?, 'SYM_FRAYED_GILLS', 'PRESENT')", (case_id,))
        cursor.execute("""
            INSERT INTO water_measurements (case_id, temperature, ph, dissolved_oxygen, ammonia, nitrite)
            VALUES (?, 29.0, 7.2, 2.4, 0.60, 0.05)
        """, (case_id,))
        
        ev_json = json.dumps({
            "inferred_disease": "Bacterial Gill Diseases",
            "inferred_disease_vietnamese": "Bệnh thối mang do vi khuẩn",
            "explanation": "ĐỒNG THUẬN CHỨNG CỨ: Kết quả nhận diện AI hoàn toàn phù hợp với triệu chứng tơ mang xơ xác và môi trường thiếu oxy.",
            "matched_rules": ["Rule_BGD_LowOxygen", "Rule_BGD_HighAmmonia"],
            "score_type": "Rule-based Decision Support Score"
        }, ensure_ascii=False)
        conf_json = json.dumps({"conflict_detected": False, "conflict_type": "NONE"}, ensure_ascii=False)
        
        cursor.execute("""
            INSERT INTO reasoning_results (case_id, disease_uri, support_score, evidence_json, conflict_json, ontology_version)
            VALUES (?, 'http://www.semanticweb.org/fish-disease-ontology#Bacterial_Gill_Disease', 0.81, ?, ?, '2.0.0 (W3C OWL 2)')
        """, (case_id, ev_json, conf_json))
        res_id = cursor.lastrowid
        
        rec_text = "PHÁC ĐỒ ĐIỀU TRỊ:\n- Kháng sinh Oxytetracycline 50-75mg/kg trong 7 ngày.\n- Sát trùng nguồn nước bằng Iodine 90 (0.4 ppm).\n\nXỬ LÝ NƯỚC CẤP THỜI:\n- Bật quạt khí sục oxy ngay lập tức để nâng DO > 4.0 mg/L."
        cursor.execute("""
            INSERT INTO recommendations (result_id, case_id, disease_uri, recommendation_text, source_note)
            VALUES (?, ?, 'http://www.semanticweb.org/fish-disease-ontology#Bacterial_Gill_Disease', ?, 'Chuẩn WOAH & FAO')
        """, (res_id, case_id, rec_text))
        print(f"✓ Đã nạp ca chẩn đoán mẫu khởi đầu (Case #{case_id}).")

    conn.commit()
    conn.close()
    print("=" * 60)
    print("🎉 KHỞI TẠO CƠ SỞ DỮ LIỆU FishDiagnosis.db HOÀN TẤT VÀ SẴN SÀNG!")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Khởi tạo CSDL FishDiagnosis.db")
    parser.add_argument("--force", action="store_true", help="Xóa CSDL cũ và tạo mới hoàn toàn")
    args = parser.parse_args()
    init_database(force=args.force)
