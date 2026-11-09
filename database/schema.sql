-- ==============================================================================
-- CƠ SỞ DỮ LIỆU: FishDiagnosis.db
-- Đề tài: Hệ thống hỗ trợ phát hiện bệnh cá dựa trên học sâu và ontology
-- Sinh viên: Trương Thị Kim Yên (MSSV: 65134516) - Khoa CNTT, ĐH Nha Trang
-- Hệ quản trị CSDL: SQLite 3
-- ==============================================================================

PRAGMA foreign_keys = ON;

-- 1. BẢNG NGƯỜI DÙNG HỆ THỐNG
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'Kỹ thuật viên Thủy sản',
    email TEXT,
    password_hash TEXT
);

-- 2. BẢNG HỒ SƠ CA CHẨN ĐOÁN
CREATE TABLE IF NOT EXISTS diagnostic_cases (
    case_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    fish_species_uri TEXT,
    note TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- 3. BẢNG ẢNH ĐẦU VÀO CA BỆNH
CREATE TABLE IF NOT EXISTS case_images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    original_name TEXT,
    FOREIGN KEY (case_id) REFERENCES diagnostic_cases(case_id) ON DELETE CASCADE
);

-- 4. BẢNG PHIÊN BẢN MÔ HÌNH HỌC SÂU (MODEL RUNS)
CREATE TABLE IF NOT EXISTS model_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    version TEXT NOT NULL,
    metrics_json TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 5. BẢNG KẾT QUẢ DỰ ĐOÁN THỊ GIÁC AI
CREATE TABLE IF NOT EXISTS ai_predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id INTEGER,
    run_id INTEGER NOT NULL,
    case_id INTEGER NOT NULL,
    predicted_label TEXT NOT NULL,
    confidence_score REAL NOT NULL,
    topk_json TEXT, -- Lưu Top-k prediction dưới dạng JSON [{"class_name": "...", "confidence": ...}]
    FOREIGN KEY (image_id) REFERENCES case_images(image_id) ON DELETE SET NULL,
    FOREIGN KEY (run_id) REFERENCES model_runs(run_id),
    FOREIGN KEY (case_id) REFERENCES diagnostic_cases(case_id) ON DELETE CASCADE
);

-- 6. BẢNG DANH MỤC TRIỆU CHỨNG LÂM SÀNG CHUẨN
CREATE TABLE IF NOT EXISTS symptoms (
    symptom_code TEXT PRIMARY KEY,
    symptom_name TEXT NOT NULL,
    symptom_uri TEXT NOT NULL
);

-- 7. BẢNG TRIỆU CHỨNG GHI NHẬN THEO CA BỆNH
CREATE TABLE IF NOT EXISTS case_symptoms (
    case_id INTEGER NOT NULL,
    symptom_code TEXT NOT NULL,
    selected_value TEXT DEFAULT 'PRESENT',
    PRIMARY KEY (case_id, symptom_code),
    FOREIGN KEY (case_id) REFERENCES diagnostic_cases(case_id) ON DELETE CASCADE,
    FOREIGN KEY (symptom_code) REFERENCES symptoms(symptom_code)
);

-- 8. BẢNG THÔNG SỐ CHẤT LƯỢNG MÔI TRƯỜNG NƯỚC AO NUÔI
CREATE TABLE IF NOT EXISTS water_measurements (
    measurement_id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER NOT NULL UNIQUE,
    temperature REAL,       -- Nhiệt độ nước (°C)
    ph REAL,                -- Độ pH nước
    dissolved_oxygen REAL,  -- Oxy hòa tan (DO - mg/L)
    ammonia REAL,           -- Khí độc Amoniac tổng (TAN - mg/L)
    nitrite REAL,           -- Khí độc Nitrite (NO2- - mg/L)
    alkalinity REAL,        -- Độ kiềm (mg CaCO3/L)
    turbidity REAL,         -- Độ đục / Độ trong (Đĩa Secchi - cm)
    FOREIGN KEY (case_id) REFERENCES diagnostic_cases(case_id) ON DELETE CASCADE
);

-- 9. BẢNG KẾT QUẢ SUY LUẬN TRI THỨC ONTOLOGY & HỖ TRỢ QUYẾT ĐỊNH
CREATE TABLE IF NOT EXISTS reasoning_results (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER NOT NULL UNIQUE,
    disease_uri TEXT NOT NULL,
    support_score REAL NOT NULL, -- Hybrid Support Score [0.0 - 1.0]
    evidence_json TEXT,          -- JSON lưu bằng chứng ủng hộ, luật Ontology kích hoạt, Grad-CAM path
    conflict_json TEXT,          -- JSON lưu cờ mâu thuẫn (conflict_detected, conflict_type)
    ontology_version TEXT NOT NULL,
    FOREIGN KEY (case_id) REFERENCES diagnostic_cases(case_id) ON DELETE CASCADE
);

-- 10. BẢNG KHUYẾN NGHỊ PHÁC ĐỒ ĐIỀU TRỊ VÀ PHÒNG NGỪA (WOAH / FAO)
CREATE TABLE IF NOT EXISTS recommendations (
    recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id INTEGER NOT NULL,
    case_id INTEGER NOT NULL,
    disease_uri TEXT NOT NULL,
    recommendation_text TEXT NOT NULL,
    source_note TEXT,
    FOREIGN KEY (result_id) REFERENCES reasoning_results(result_id) ON DELETE CASCADE,
    FOREIGN KEY (case_id) REFERENCES diagnostic_cases(case_id) ON DELETE CASCADE
);

-- CÁC CHỈ MỤC TỐI ƯU HÓA TRUY VẤN
CREATE INDEX IF NOT EXISTS idx_cases_created ON diagnostic_cases(created_at);
CREATE INDEX IF NOT EXISTS idx_ai_predictions_case ON ai_predictions(case_id);
CREATE INDEX IF NOT EXISTS idx_reasoning_case ON reasoning_results(case_id);
