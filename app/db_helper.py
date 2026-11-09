# -*- coding: utf-8 -*-
import os
import sys
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent

def get_db_path() -> Path:
    p_lower = BASE_DIR / "database" / "FishDiagnosis.db"
    p_upper = BASE_DIR / "Database" / "FishDiagnosis.db"
    if p_lower.exists():
        return p_lower
    if p_upper.exists():
        return p_upper
    return p_lower

DB_PATH = get_db_path()
UPLOADS_DIR = BASE_DIR / "app" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(get_db_path()))
    conn.row_factory = sqlite3.Row
    return conn

def init_master_data():
    """Khởi tạo dữ liệu nền tảng cho CSDL (triệu chứng chuẩn và mô hình EfficientNet-B0)."""
    conn = get_connection()
    c = conn.cursor()
    
    # 1. Đảm bảo có User mặc định
    c.execute("SELECT user_id FROM users WHERE user_id = 1;")
    if not c.fetchone():
        c.execute("""
            INSERT INTO users (user_id, full_name, role, email) 
            VALUES (1, 'Kỹ thuật viên Trương Thị Kim Yên', 'Kỹ thuật viên Thủy sản', 'kimyen@ntu.edu.vn')
        """)
    else:
        # Cập nhật tên kỹ thuật viên cho chuyên nghiệp
        c.execute("""
            UPDATE users SET full_name = 'Kỹ thuật viên Trương Thị Kim Yên', role = 'Kỹ thuật viên Thủy sản' WHERE user_id = 1;
        """)

    # 2. Khởi tạo danh mục triệu chứng chuẩn nếu chưa có
    standard_symptoms = [
        ("SYM_FRAYED_GILLS", "Mang xơ xác, hoại tử tơ mang", "http://www.semanticweb.org/fish-disease-ontology#Frayed_Gills"),
        ("SYM_SKIN_ULCERS", "Loét da, xuất huyết thân và gốc vây", "http://www.semanticweb.org/fish-disease-ontology#Skin_Ulcers"),
        ("SYM_ABDOMINAL_DISTENSION", "Chướng bụng, sưng đỏ hậu môn", "http://www.semanticweb.org/fish-disease-ontology#Abdominal_Distension"),
        ("SYM_LETHARGY", "Bơi lờ đờ, nổi đầu tách đàn", "http://www.semanticweb.org/fish-disease-ontology#Lethargy_Anorexia"),
        ("SYM_NORMAL", "Ngoại hình bình thường, bơi lội nhanh nhẹn", "http://www.semanticweb.org/fish-disease-ontology#Normal_Behavior_Appearance"),
    ]
    for code, name, uri in standard_symptoms:
        c.execute("""
            INSERT OR IGNORE INTO symptoms (symptom_code, symptom_name, symptom_uri)
            VALUES (?, ?, ?)
        """, (code, name, uri))

    # 3. Đảm bảo có bản ghi model_runs cho mô hình chính thức EfficientNet-B0
    c.execute("SELECT run_id FROM model_runs WHERE model_name = 'EfficientNet-B0';")
    if not c.fetchone():
        metrics_dict = {
            "checkpoint": "models/efficientnet_b0_best.pth",
            "parameters_m": 4.01,
            "test_accuracy": 0.8267,
            "test_macro_f1": 0.8512,
            "test_macro_precision": 0.8743,
            "test_macro_recall": 0.8527
        }
        c.execute("""
            INSERT INTO model_runs (model_name, version, metrics_json, created_at)
            VALUES (?, ?, ?, ?)
        """, ("EfficientNet-B0", "v1.0 (Official Best Checkpoint)", json.dumps(metrics_dict, ensure_ascii=False), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    conn.commit()
    conn.close()

def get_official_model_run_id() -> int:
    """Lấy run_id của EfficientNet-B0."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT run_id FROM model_runs WHERE model_name = 'EfficientNet-B0' ORDER BY run_id DESC LIMIT 1;")
    row = c.fetchone()
    conn.close()
    return row["run_id"] if row else 1

def save_full_diagnostic_case(
    original_image_path: str,
    original_filename: str,
    gradcam_image_path: Optional[str],
    fish_species: str,
    note: str,
    dl_results: Dict[str, Any],
    symptoms_selected: List[str],
    water_data: Dict[str, Optional[float]],
    reasoning_res: Dict[str, Any],
    recommendations_data: Dict[str, Any],
    user_id: int = 1
) -> int:
    """
    Lưu toàn bộ hồ sơ ca chẩn đoán lâm sàng xuống CSDL SQLite (bảo đảm tính toàn vẹn và liên kết khóa ngoại).
    Trả về: case_id vừa tạo.
    """
    conn = get_connection()
    c = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        # 1. Bảng diagnostic_cases
        c.execute("""
            INSERT INTO diagnostic_cases (user_id, created_at, fish_species_uri, note)
            VALUES (?, ?, ?, ?)
        """, (user_id, now_str, fish_species, note))
        case_id = c.lastrowid

        # 2. Bảng case_images (lưu ảnh gốc)
        c.execute("""
            INSERT INTO case_images (case_id, file_path, original_name)
            VALUES (?, ?, ?)
        """, (case_id, original_image_path, original_filename))
        image_id = c.lastrowid

        # 3. Bảng ai_predictions
        run_id = get_official_model_run_id()
        top1_label = dl_results.get("top1_predicted_class", "Unknown")
        top1_conf = float(dl_results.get("top1_confidence", 0.0))
        top3_json = json.dumps(dl_results.get("top3_predictions", []), ensure_ascii=False)

        c.execute("""
            INSERT INTO ai_predictions (image_id, run_id, case_id, predicted_label, confidence_score, topk_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (image_id, run_id, case_id, top1_label, top1_conf, top3_json))

        # 4. Bảng case_symptoms (lưu danh mục triệu chứng người dùng chọn)
        for sym_code in symptoms_selected:
            c.execute("""
                INSERT INTO case_symptoms (case_id, symptom_code, selected_value)
                VALUES (?, ?, 'PRESENT')
            """, (case_id, sym_code))

        # 5. Bảng water_measurements (lưu từng trường thông số riêng biệt)
        temp = water_data.get("temperature")
        ph = water_data.get("ph")
        do_val = water_data.get("dissolved_oxygen")
        nh3 = water_data.get("ammonia")
        no2 = water_data.get("nitrite")
        c.execute("""
            INSERT INTO water_measurements (case_id, temperature, ph, dissolved_oxygen, ammonia, nitrite, alkalinity, turbidity)
            VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)
        """, (case_id, temp, ph, do_val, nh3, no2))

        # 6. Bảng reasoning_results
        disease_uri = reasoning_res.get("inferred_disease_uri", "")
        support_score = float(reasoning_res.get("hybrid_support_score", 0.0))
        
        evidence_dict = {
            "inferred_disease": reasoning_res.get("inferred_disease"),
            "inferred_disease_vietnamese": reasoning_res.get("inferred_disease_vietnamese"),
            "explanation": reasoning_res.get("explanation"),
            "matched_rules": reasoning_res.get("matched_ontology_rules", []),
            "differential_diagnosis": reasoning_res.get("differential_diagnosis", []),
            "gradcam_image_path": gradcam_image_path,
            "score_type": reasoning_res.get("score_type", "Rule-based Decision Support Score")
        }
        conflict_dict = {
            "conflict_detected": reasoning_res.get("conflict_detected", False),
            "conflict_type": reasoning_res.get("conflict_type", "NONE")
        }

        c.execute("""
            INSERT INTO reasoning_results (case_id, disease_uri, support_score, evidence_json, conflict_json, ontology_version)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (case_id, disease_uri, support_score, json.dumps(evidence_dict, ensure_ascii=False), json.dumps(conflict_dict, ensure_ascii=False), "2.0.0 (W3C OWL 2)"))
        result_id = c.lastrowid

        # 7. Bảng recommendations
        treatments = recommendations_data.get("treatments", [])
        prevention = recommendations_data.get("prevention", "")
        water_adjustments = recommendations_data.get("water_adjustments", [])
        
        full_rec_text = "PHÁC ĐỒ ĐIỀU TRỊ:\n" + "\n".join(f"- {t}" for t in treatments)
        if prevention:
            full_rec_text += f"\n\nBIỆN PHÁP PHÒNG NGỪA:\n- {prevention}"
        if water_adjustments:
            full_rec_text += "\n\nXỬ LÝ MÔI TRƯỜNG NƯỚC CẤP THỜI:\n" + "\n".join(f"- {w}" for w in water_adjustments)

        c.execute("""
            INSERT INTO recommendations (result_id, case_id, disease_uri, recommendation_text, source_note)
            VALUES (?, ?, ?, ?, ?)
        """, (result_id, case_id, disease_uri, full_rec_text, "Hướng dẫn Thú y Thủy sản WOAH (2021) & FAO Tech Paper 402"))

        conn.commit()
        return case_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def get_diagnostic_history(limit: int = 100, disease_filter: Optional[str] = None, conflict_only: bool = False) -> List[Dict[str, Any]]:
    """Truy vấn danh sách lịch sử các ca chẩn đoán để hiển thị trên bảng."""
    conn = get_connection()
    c = conn.cursor()
    
    query = """
        SELECT 
            dc.case_id,
            dc.created_at,
            dc.fish_species_uri AS species,
            ci.file_path AS image_path,
            ap.predicted_label AS ai_predicted_label,
            ap.confidence_score AS ai_confidence,
            rr.support_score,
            rr.disease_uri,
            rr.conflict_json,
            rr.evidence_json,
            rr.ontology_version,
            mr.model_name,
            mr.version AS model_version
        FROM diagnostic_cases dc
        LEFT JOIN case_images ci ON dc.case_id = ci.case_id
        LEFT JOIN ai_predictions ap ON dc.case_id = ap.case_id
        LEFT JOIN reasoning_results rr ON dc.case_id = rr.case_id
        LEFT JOIN model_runs mr ON ap.run_id = mr.run_id
        ORDER BY dc.case_id DESC
        LIMIT ?;
    """
    c.execute(query, (limit,))
    rows = c.fetchall()
    conn.close()
    
    results = []
    for r in rows:
        conf_data = {}
        if r["conflict_json"]:
            try:
                conf_data = json.loads(r["conflict_json"])
            except Exception:
                pass
                
        ev_data = {}
        if r["evidence_json"]:
            try:
                ev_data = json.loads(r["evidence_json"])
            except Exception:
                pass

        final_disease = ev_data.get("inferred_disease_vietnamese") or ev_data.get("inferred_disease") or r["ai_predicted_label"] or "Chưa rõ"
        is_conflict = conf_data.get("conflict_detected", False)

        if conflict_only and not is_conflict:
            continue
        if disease_filter and disease_filter != "Tất cả" and disease_filter not in str(final_disease):
            continue

        results.append({
            "case_id": r["case_id"],
            "created_at": r["created_at"],
            "species": r["species"] or "Không xác định",
            "image_path": r["image_path"],
            "ai_label": r["ai_predicted_label"],
            "ai_conf": r["ai_confidence"],
            "final_diagnosis": final_disease,
            "support_score": r["support_score"],
            "is_conflict": is_conflict,
            "conflict_type": conf_data.get("conflict_type", "NONE"),
            "model_version": f"{r['model_name']} ({r['model_version']})" if r["model_name"] else "EfficientNet-B0",
            "ontology_version": r["ontology_version"] or "2.0.0"
        })
        
    return results

def get_case_detail(case_id: int) -> Optional[Dict[str, Any]]:
    """Truy vết toàn diện hồ sơ một ca chẩn đoán theo case_id."""
    conn = get_connection()
    c = conn.cursor()
    
    # 1. Thông tin chung ca bệnh
    c.execute("""
        SELECT dc.*, u.full_name, u.role 
        FROM diagnostic_cases dc
        LEFT JOIN users u ON dc.user_id = u.user_id
        WHERE dc.case_id = ?;
    """, (case_id,))
    case_info = c.fetchone()
    if not case_info:
        conn.close()
        return None

    # 2. Ảnh ca bệnh
    c.execute("SELECT * FROM case_images WHERE case_id = ? ORDER BY image_id ASC;", (case_id,))
    images = [dict(r) for r in c.fetchall()]

    # 3. Kết quả AI và Model
    c.execute("""
        SELECT ap.*, mr.model_name, mr.version AS model_version, mr.metrics_json
        FROM ai_predictions ap
        LEFT JOIN model_runs mr ON ap.run_id = mr.run_id
        WHERE ap.case_id = ?;
    """, (case_id,))
    ai_info = c.fetchone()

    # 4. Triệu chứng lâm sàng
    c.execute("""
        SELECT cs.*, s.symptom_name, s.symptom_uri
        FROM case_symptoms cs
        LEFT JOIN symptoms s ON cs.symptom_code = s.symptom_code
        WHERE cs.case_id = ?;
    """, (case_id,))
    symptoms = [dict(r) for r in c.fetchall()]

    # 5. Thông số môi trường nước
    c.execute("SELECT * FROM water_measurements WHERE case_id = ?;", (case_id,))
    water = c.fetchone()

    # 6. Kết quả suy luận Ontology
    c.execute("SELECT * FROM reasoning_results WHERE case_id = ?;", (case_id,))
    reasoning = c.fetchone()

    # 7. Khuyến nghị điều trị
    c.execute("SELECT * FROM recommendations WHERE case_id = ?;", (case_id,))
    recs = [dict(r) for r in c.fetchall()]

    conn.close()

    # Xử lý parse JSON
    evidence_data = {}
    conflict_data = {}
    if reasoning:
        if reasoning["evidence_json"]:
            try: evidence_data = json.loads(reasoning["evidence_json"])
            except Exception: pass
        if reasoning["conflict_json"]:
            try: conflict_data = json.loads(reasoning["conflict_json"])
            except Exception: pass

    topk_list = []
    if ai_info and ai_info["topk_json"]:
        try: topk_list = json.loads(ai_info["topk_json"])
        except Exception: pass

    return {
        "case_info": dict(case_info),
        "images": images,
        "ai_prediction": dict(ai_info) if ai_info else {},
        "top3_predictions": topk_list,
        "symptoms": symptoms,
        "water_measurements": dict(water) if water else {},
        "reasoning": dict(reasoning) if reasoning else {},
        "evidence": evidence_data,
        "conflict": conflict_data,
        "recommendations": recs,
        "gradcam_path": evidence_data.get("gradcam_image_path")
    }

# Tự động khởi tạo dữ liệu chuẩn khi module được import
init_master_data()
