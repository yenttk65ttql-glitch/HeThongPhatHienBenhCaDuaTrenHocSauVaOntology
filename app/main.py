# -*- coding: utf-8 -*-
"""
HỆ THỐNG HỖ TRỢ CHẨN ĐOÁN BỆNH CÁ NƯỚC NGỌT DỰA TRÊN HỌC SÂU VÀ ONTOLOGY
Đề tài Tốt nghiệp - Khoa Công Nghệ Thông Tin - Đại học Nha Trang
Sinh viên thực hiện: Trương Thị Kim Yên (MSSV: 65134516 - Lớp: 65.TTQL-1)
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from PIL import Image

import streamlit as st

# Thiết lập cấu hình trang Streamlit
st.set_page_config(
    page_title="FishDx AI-Ontology | Chẩn Đoán Bệnh Cá",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Thêm đường dẫn project_root vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_engine import FishDiseaseHybridPipeline
from app import db_helper

UPLOADS_DIR = PROJECT_ROOT / "app" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Cache khởi tạo Pipeline (chỉ nạp một lần duy nhất vào bộ nhớ)
@st.cache_resource(show_spinner="Đang khởi tạo Pipeline AI (EfficientNet-B0) và nạp tri thức Ontology (W3C OWL 2)...")
def get_hybrid_pipeline():
    return FishDiseaseHybridPipeline()

# Danh mục ảnh mẫu có sẵn trong tập Test để demo nhanh
SAMPLE_IMAGES = {
    "Chọn ảnh mẫu thử nghiệm nhanh...": None,
    "Mẫu 1 - Bệnh thối mang (BGD) [Fish_19]": "data/raw/Bacterial Gill Diseases/Bacterial Gill Diseases/Silver_carp_(Hypophthalmichthys molitrix)/Fish_19/IMG20260206071503.jpg",
    "Mẫu 2 - Bệnh đốm đỏ (BRD) [Fish_55]": "data/raw/Bacterial Red Diseases/Bacterial Red Diseases/Silver_carp_(Hypophthalmichthys molitrix)/Fish_55/IMG20260206071236.jpg",
    "Mẫu 3 - Hội chứng lở loét (EUS) [Fish_75]": "data/raw/Epizootic Ulcerative Syndrome (EUS)/Epizootic Ulcerative Syndrome (EUS)/Silver_carp_(Hypophthalmichthys molitrix)/Fish_75/IMG20260212145352.jpg",
    "Mẫu 4 - Cá khỏe mạnh (Healthy) [Fish_84]": "data/raw/Healthy Fish/Healthy Fish/Silver_carp_(Hypophthalmichthys molitrix)/Fish_84/IMG20260315103620.jpg"
}

# Danh mục triệu chứng lâm sàng chuẩn
SYMPTOM_OPTIONS = {
    "SYM_FRAYED_GILLS": "Mang xơ xác, hoại tử phiến mang, dính nhiều nhớt bẩn",
    "SYM_SKIN_ULCERS": "Vết loét da, xuất huyết đỏ ở thân và gốc vây",
    "SYM_ABDOMINAL_DISTENSION": "Chướng bụng, hậu môn sưng đỏ lồi ra ngoài",
    "SYM_LETHARGY": "Bơi lờ đờ, mất định hướng, nổi đầu tách đàn ở tầng mặt",
    "SYM_NORMAL": "Ngoại hình bình thường, da vây nguyên vẹn, phản xạ nhanh"
}

# ==============================================================================
# SIDEBAR - THÔNG TIN HỆ THỐNG
# ==============================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=70)
    st.title("FishDx System")
    st.markdown("**Hệ thống Hỗ trợ Chẩn đoán Bệnh Cá Nước Ngọt**")
    st.caption("Kết hợp Học sâu (EfficientNet-B0) & Ontology Tri thức (OWL 2)")
    st.markdown("---")
    
    st.markdown("### 📌 Thông tin Đồ án")
    st.markdown("- **Sinh viên:** Trương Thị Kim Yên")
    st.markdown("- **MSSV:** 65134516 | **Lớp:** 65.TTQL-1")
    st.markdown("- **Trường:** Đại học Nha Trang (NTU)")
    st.markdown("- **Khoa:** Công nghệ Thông tin")
    
    st.markdown("---")
    st.markdown("### ⚙️ Thông số Kỹ thuật")
    st.info("""
    - **Vision AI:** EfficientNet-B0 (4.01M params)
    - **Test Macro-F1:** 85.12% | **Accuracy:** 82.67%
    - **Ontology:** FishDiseaseOntology.owl (v2.0)
    - **Reasoner:** HermiT + SPARQL Execution Engine
    - **Explainability:** Grad-CAM (Target Layer: `features[-1]`)
    - **Database:** SQLite3 (`Database/FishDiagnosis.db`)
    """)

# ==============================================================================
# TIÊU ĐỀ CHÍNH
# ==============================================================================
st.markdown("<h2 style='text-align: center; color: #1E3A8A;'>🐟 HỆ THỐNG HỖ TRỢ CHẨN ĐOÁN BỆNH CÁ NƯỚC NGỌT</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #4B5563;'>Mô hình Suy luận Lai Định Hướng Tri Thức (Ontology-Driven Hybrid AI Reasoning with Grad-CAM XAI)</p>", unsafe_allow_html=True)

# TẠO TABS CHÍNH
tab1, tab2, tab3 = st.tabs([
    "🔬 Chẩn Đoán Ca Bệnh Mới", 
    "📜 Lịch Sử Chẩn Đoán & Truy Vết", 
    "ℹ️ Kiến Trúc Hệ Thống & Tri Thức"
])

# ==============================================================================
# TAB 1: CHẨN ĐOÁN CA BỆNH MỚI
# ==============================================================================
with tab1:
    col_input, col_output = st.columns([1, 1.2], gap="large")
    
    # --------------------------------------------------------------------------
    # CỘT TRÁI: TIẾP NHẬN DỮ LIỆU ĐẦU VÀO
    # --------------------------------------------------------------------------
    with col_input:
        st.subheader("1. Thu Thập Dữ Liệu Đầu Vào")
        
        # 1.1. Tải ảnh hoặc chọn ảnh mẫu
        st.markdown("##### 📷 Ảnh chụp cá nghi nhiễm bệnh")
        uploaded_file = st.file_uploader("Tải tệp ảnh cá lên (JPG, PNG)", type=["jpg", "jpeg", "png"])
        sample_choice = st.selectbox("Hoặc chọn ảnh mẫu kiểm thử:", list(SAMPLE_IMAGES.keys()))
        
        selected_image_path = None
        input_pil_img = None
        orig_filename = "uploaded_fish.jpg"
        
        if uploaded_file is not None:
            input_pil_img = Image.open(uploaded_file).convert("RGB")
            orig_filename = uploaded_file.name
            st.image(input_pil_img, caption=f"Ảnh người dùng tải lên: {orig_filename}", use_container_width=True)
        elif sample_choice and SAMPLE_IMAGES[sample_choice]:
            sample_rel_path = SAMPLE_IMAGES[sample_choice]
            selected_image_path = str(PROJECT_ROOT / sample_rel_path)
            if os.path.exists(selected_image_path):
                input_pil_img = Image.open(selected_image_path).convert("RGB")
                orig_filename = Path(selected_image_path).name
                st.image(input_pil_img, caption=f"Ảnh mẫu kiểm thử: {orig_filename}", use_container_width=True)
                
        # 1.2. Thông tin cá thể & Ngữ cảnh ao nuôi
        st.markdown("##### 🐟 Thông tin cá thể & Ngữ cảnh")
        fish_species = st.selectbox(
            "Loài cá nuôi:",
            [
                "Cá mè trắng (Hypophthalmichthys molitrix)",
                "Cá trôi Ấn Độ (Labeo rohita)",
                "Cá trắm cỏ (Ctenopharyngodon idella)",
                "Khác / Chưa xác định"
            ]
        )
        context_note = st.text_input("Ghi chú ao nuôi / Vị trí lồng bè:", placeholder="Ví dụ: Ao số 3 - Cá có dấu hiệu nổi đầu sáng sớm...")

        # 1.3. Nhập thông số môi trường nước (TỪNG TRƯỜNG RIÊNG BIỆT)
        st.markdown("##### 🧪 Chỉ số chất lượng môi trường nước ao")
        st.caption("Các thông số được kiểm soát ranh giới theo chuẩn Datatype Facets trong Ontology")
        
        skip_water = st.checkbox("Chưa có máy đo môi trường (Bỏ qua số liệu nước)", value=False)
        
        water_measurements = {}
        if not skip_water:
            c_w1, c_w2 = st.columns(2)
            with c_w1:
                val_temp = st.number_input("Nhiệt độ nước (°C):", min_value=0.0, max_value=45.0, value=28.0, step=0.5, help="Quy luật EUS kích hoạt khi T < 25°C")
                val_do = st.number_input("Oxy hòa tan (DO - mg/L):", min_value=0.0, max_value=20.0, value=4.5, step=0.1, help="Quy luật BGD kích hoạt khi DO < 3.5 mg/L")
            with c_w2:
                val_ph = st.number_input("Độ pH nước:", min_value=0.0, max_value=14.0, value=7.2, step=0.1, help="Quy luật EUS kích hoạt khi pH < 6.5")
                val_nh3 = st.number_input("Khí độc Amoniac (TAN - mg/L):", min_value=0.0, max_value=10.0, value=0.20, step=0.05, help="Rủi ro cao khi TAN > 0.5 mg/L")
                
            val_no2 = st.number_input("Khí độc Nitrite (NO2- - mg/L):", min_value=0.0, max_value=5.0, value=0.05, step=0.01)
            
            water_measurements = {
                "temperature": float(val_temp),
                "ph": float(val_ph),
                "dissolved_oxygen": float(val_do),
                "ammonia": float(val_nh3),
                "nitrite": float(val_no2)
            }
        else:
            water_measurements = {}

        # 1.4. Nhập triệu chứng lâm sàng (TỪNG TRƯỜNG CHỌN RIÊNG BIỆT)
        st.markdown("##### 🩺 Triệu chứng lâm sàng quan sát được")
        selected_symptoms = []
        for code, label in SYMPTOM_OPTIONS.items():
            if st.checkbox(f"{label}", key=f"chk_{code}", value=(code == "SYM_NORMAL" if skip_water and not uploaded_file else False)):
                selected_symptoms.append(code)
                
        # Nút kích hoạt chẩn đoán
        st.markdown("---")
        btn_diagnose = st.button("🚀 BẮT ĐẦU CHẨN ĐOÁN TOÀN DIỆN", type="primary", use_container_width=True)

    # --------------------------------------------------------------------------
    # CỘT PHẢI: KẾT QUẢ CHẨN ĐOÁN & GIẢI THÍCH GRAD-CAM
    # --------------------------------------------------------------------------
    with col_output:
        st.subheader("2. Kết Quả Chẩn Đoán & Khuyến Nghị")
        
        if btn_diagnose:
            if input_pil_img is None:
                st.error("⚠️ Vui lòng tải ảnh cá lên hoặc chọn một ảnh mẫu thử nghiệm trước khi chẩn đoán!")
            else:
                with st.spinner("Đang thực thi Pipeline: Dự đoán EfficientNet-B0 ➔ Grad-CAM ➔ Truy vấn SPARQL ➔ Phân giải mâu thuẫn..."):
                    pipeline = get_hybrid_pipeline()
                    
                    # Lưu ảnh tạm thời
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    temp_img_name = f"case_{timestamp_str}_{orig_filename}"
                    temp_img_path = UPLOADS_DIR / temp_img_name
                    input_pil_img.save(temp_img_path)
                    
                    gradcam_img_name = f"gradcam_{timestamp_str}_{orig_filename}.png"
                    gradcam_save_path = UPLOADS_DIR / gradcam_img_name
                    
                    # Thực thi pipeline chẩn đoán
                    diag_result = pipeline.diagnose(
                        image_input=input_pil_img,
                        observed_symptoms=selected_symptoms,
                        water_measurements=water_measurements,
                        generate_gradcam=True,
                        gradcam_save_path=gradcam_save_path
                    )
                    
                    # Lưu toàn bộ ca bệnh vào CSDL SQLite
                    dl_res = diag_result["deep_learning_results"]
                    onto_res = diag_result["ontology_hybrid_conclusion"]
                    recs_res = diag_result["actionable_recommendations"]
                    
                    new_case_id = db_helper.save_full_diagnostic_case(
                        original_image_path=f"app/uploads/{temp_img_name}",
                        original_filename=orig_filename,
                        gradcam_image_path=f"app/uploads/{gradcam_img_name}",
                        fish_species=fish_species,
                        note=context_note,
                        dl_results=dl_res,
                        symptoms_selected=selected_symptoms,
                        water_data=water_measurements,
                        reasoning_res=onto_res,
                        recommendations_data=recs_res,
                        user_id=1
                    )
                    
                    st.success(f"✅ Chẩn đoán thành công và đã lưu vết vào CSDL! (Mã hồ sơ: **Case #{new_case_id}**)")
                    
                    # --- KHỐI A: KẾT QUẢ AI & GRAD-CAM ---
                    st.markdown("#### A. Nhận diện Thị giác Máy tính (EfficientNet-B0)")
                    top1_name = dl_res["top1_predicted_class"]
                    top1_conf = dl_res["top1_confidence"]
                    
                    st.write(f"**Dự đoán hàng đầu (Top-1):** `{top1_name}` — Độ tin cậy: **{top1_conf:.1%}**")
                    
                    # Hiển thị Top-3 Bar chart
                    for p in dl_res["top3_predictions"]:
                        st.progress(float(p["confidence"]), text=f"{p['class_name']}: {p['confidence']:.1%}")
                        
                    # Hiển thị ảnh Grad-CAM 3 khung hình
                    if os.path.exists(gradcam_save_path):
                        st.image(str(gradcam_save_path), caption="Giải thích vùng tổn thương với Grad-CAM (Target layer: features[-1])", use_container_width=True)

                    # --- KHỐI B: SUY LUẬN ONTOLOGY & PHÁT HIỆN MÂU THUẪN ---
                    st.markdown("#### B. Kết luận Suy luận Tri thức (AI–Ontology)")
                    final_disease = onto_res["inferred_disease_vietnamese"]
                    support_score = onto_res["hybrid_support_score"]
                    is_conflict = onto_res["conflict_detected"]
                    conflict_type = onto_res["conflict_type"]
                    
                    # Thẻ hiển thị mâu thuẫn hoặc đồng thuận
                    if is_conflict:
                        st.warning(f"⚠️ **CẢNH BÁO MÂU THUẪN ĐƯỢC PHÁT HIỆN! ({conflict_type})**\n\n{onto_res['explanation']}")
                    else:
                        st.info(f"💡 **TRẠNG THÁI ĐỒNG THUẬN CHỨNG CỨ:**\n\n{onto_res['explanation']}")
                        
                    st.metric(
                        label="Chẩn đoán cuối cùng được hỗ trợ:",
                        value=final_disease,
                        delta=f"Điểm HSS: {support_score:.2f} ({onto_res['score_type']})"
                    )
                    
                    # Bảng chẩn đoán phân biệt
                    with st.expander("📊 Xem bảng Chẩn đoán Phân biệt (Differential Diagnosis Ranking)"):
                        st.table([
                            {
                                "Bệnh ứng viên": d["vietnamese_name"],
                                "Điểm tổng hợp (HSS)": f"{d['composite_score']:.4f}",
                                "Độ khớp triệu chứng": f"{d['symptom_match_rate']:.1%}",
                                "Rủi ro môi trường": f"{d['environmental_risk']:.2f}"
                            }
                            for d in onto_res.get("differential_diagnosis", [])
                        ])
                        
                    # Danh mục luật Ontology được kích hoạt
                    if onto_res.get("matched_ontology_rules"):
                        st.caption(f"Quy tắc Ontology kích hoạt: `{', '.join(onto_res['matched_ontology_rules'])}`")

                    # --- KHỐI C: PHÁC ĐỒ ĐIỀU TRỊ KHUYẾN NGHỊ ---
                    st.markdown("#### C. Khuyến Nghị Phác Đồ Điều Trị & Phòng Ngừa")
                    st.caption("Tri thức chuẩn hóa từ Tổ chức Thú y Thế giới (WOAH) và FAO")
                    
                    if recs_res.get("treatments"):
                        st.markdown("**💊 Phác đồ điều trị:**")
                        for t in recs_res["treatments"]:
                            st.markdown(f"- {t}")
                            
                    if recs_res.get("prevention"):
                        st.markdown(f"**🛡️ Biện pháp phòng ngừa:**\n- {recs_res['prevention']}")
                        
                    if recs_res.get("water_adjustments"):
                        st.markdown("**💧 Can thiệp môi trường nước khẩn cấp:**")
                        for w in recs_res["water_adjustments"]:
                            st.markdown(f"- {w}")

        else:
            st.info("👈 Hãy chọn hoặc tải ảnh cá, nhập thông số đo đạc và nhấn nút **'BẮT ĐẦU CHẨN ĐOÁN TOÀN DIỆN'** để xem kết quả.")

# ==============================================================================
# TAB 2: LỊCH SỬ CHẨN ĐOÁN & TRUY VẾT CA BỆNH
# ==============================================================================
with tab2:
    st.subheader("📜 Quản Lý Hồ Sơ & Truy Vết Lịch Sử Chẩn Đoán")
    st.caption("Dữ liệu lưu vết thực thể `DiagnosticCase` trong CSDL SQLite (`Database/FishDiagnosis.db`)")
    
    # Bộ lọc
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        disease_filter = st.selectbox(
            "Lọc theo bệnh chẩn đoán:",
            ["Tất cả", "Bệnh thối mang", "Bệnh đốm đỏ", "Hội chứng lở loét (EUS)", "Cá khỏe mạnh"]
        )
    with col_f2:
        conflict_only = st.checkbox("Chỉ xem ca có cảnh báo mâu thuẫn", value=False)
        
    # Nạp danh sách lịch sử
    history_records = db_helper.get_diagnostic_history(limit=50, disease_filter=disease_filter, conflict_only=conflict_only)
    
    if not history_records:
        st.info("Hiện chưa có hồ sơ ca bệnh nào phù hợp với bộ lọc.")
    else:
        st.write(f"Tìm thấy **{len(history_records)}** ca chẩn đoán trong cơ sở dữ liệu:")
        
        # Bảng tóm tắt
        table_rows = []
        for r in history_records:
            conflict_badge = "⚠️ Có mâu thuẫn" if r["is_conflict"] else "✅ Đồng thuận"
            table_rows.append({
                "Mã ca": f"#{r['case_id']}",
                "Thời gian": r["created_at"],
                "Loài cá": r["species"],
                "Dự đoán AI": f"{r['ai_label']} ({r['ai_conf']:.1%})" if r["ai_conf"] else (r['ai_label'] or "N/A"),
                "Chẩn đoán cuối": r["final_diagnosis"],
                "Điểm HSS": f"{r['support_score']:.2f}" if r["support_score"] else "N/A",
                "Trạng thái": conflict_badge,
                "Model & Phiên bản": r["model_version"]
            })
        st.dataframe(table_rows, use_container_width=True)
        
        # Chi tiết truy vết một ca bệnh
        st.markdown("---")
        st.markdown("#### 🔍 Truy Vết Chi Tiết Một Ca Chẩn Đoán (Audit Trail)")
        case_ids = [r["case_id"] for r in history_records]
        selected_cid = st.selectbox("Chọn mã hồ sơ để kiểm toán chi tiết:", case_ids, format_func=lambda x: f"Hồ sơ Ca bệnh #{x}")
        
        if selected_cid:
            case_detail = db_helper.get_case_detail(selected_cid)
            if case_detail:
                c_info = case_detail["case_info"]
                ai_p = case_detail["ai_prediction"]
                ev = case_detail["evidence"]
                conf = case_detail["conflict"]
                wm = case_detail["water_measurements"]
                
                # Khung thông tin truy vết phiên bản
                st.markdown(f"""
                <div style='background-color: #F3F4F6; padding: 15px; border-radius: 8px; border-left: 5px solid #3B82F6;'>
                    <b>Mã hồ sơ:</b> #{c_info['case_id']} | <b>Ngày thực hiện:</b> {c_info['created_at']} | <b>Kỹ thuật viên:</b> {c_info.get('full_name', 'Trương Thị Kim Yên')}<br>
                    <b>Mô hình Deep Learning:</b> {ai_p.get('model_name', 'EfficientNet-B0')} (Phiên bản: {ai_p.get('model_version', 'v1.0')})<br>
                    <b>Ontology Phiên bản:</b> {case_detail['reasoning'].get('ontology_version', '2.0.0')} (W3C OWL 2 Standard)
                </div>
                """, unsafe_allow_html=True)
                
                c_d1, c_d2 = st.columns([1, 1], gap="medium")
                with c_d1:
                    st.markdown("##### 🖼️ Hình ảnh & Bản đồ nhiệt Grad-CAM")
                    # Hiển thị ảnh gốc
                    if case_detail["images"]:
                        img_path = PROJECT_ROOT / case_detail["images"][0]["file_path"]
                        if img_path.exists():
                            st.image(str(img_path), caption=f"Ảnh cá gốc: {case_detail['images'][0]['original_name']}", width=280)
                            
                    # Hiển thị ảnh Grad-CAM
                    if case_detail.get("gradcam_path"):
                        g_path = PROJECT_ROOT / case_detail["gradcam_path"]
                        if g_path.exists():
                            st.image(str(g_path), caption="Giải thích vùng tổn thương Grad-CAM đã lưu vết", use_container_width=True)
                            
                with c_d2:
                    st.markdown("##### 📋 Bằng chứng Lâm sàng & Môi trường")
                    # Bảng thông số nước
                    st.markdown("**Thông số chất lượng nước ao:**")
                    st.write(f"- Nhiệt độ nước: **{wm.get('temperature', 'N/A')} °C**")
                    st.write(f"- Độ pH: **{wm.get('ph', 'N/A')}**")
                    st.write(f"- Oxy hòa tan (DO): **{wm.get('dissolved_oxygen', 'N/A')} mg/L**")
                    st.write(f"- Amoniac (TAN): **{wm.get('ammonia', 'N/A')} mg/L**")
                    st.write(f"- Nitrite (NO2-): **{wm.get('nitrite', 'N/A')} mg/L**")
                    
                    # Triệu chứng
                    st.markdown("**Triệu chứng lâm sàng ghi nhận:**")
                    if case_detail["symptoms"]:
                        for s in case_detail["symptoms"]:
                            st.markdown(f"- `{s['symptom_code']}`: {s.get('symptom_name', s['symptom_code'])}")
                    else:
                        st.write("*(Không ghi nhận triệu chứng đặc biệt)*")

                # Kết luận & Phác đồ lưu vết
                st.markdown("##### 🎯 Kết luận Suy luận & Phác đồ Đã Khuyến Nghị")
                if conf.get("conflict_detected"):
                    st.warning(f"⚠️ **Ca bệnh có mâu thuẫn ({conf.get('conflict_type')}):** {ev.get('explanation')}")
                else:
                    st.info(f"✅ **Đồng thuận:** {ev.get('explanation')}")
                    
                st.write(f"- Chẩn đoán cuối cùng: **{ev.get('inferred_disease_vietnamese', ev.get('inferred_disease', 'N/A'))}**")
                st.write(f"- Điểm hỗ trợ ra quyết định (HSS): **{case_detail['reasoning'].get('support_score', 'N/A')}**")
                if ev.get("matched_rules"):
                    st.write(f"- Các luật Ontology được kích hoạt: `{', '.join(ev['matched_rules'])}`")
                    
                if case_detail["recommendations"]:
                    st.markdown("**Phác đồ điều trị đã lưu:**")
                    st.text_area("Chi tiết khuyến nghị:", value=case_detail["recommendations"][0]["recommendation_text"], height=120, disabled=True)

# ==============================================================================
# TAB 3: KIẾN TRÚC HỆ THỐNG & CƠ SỞ TRI THỨC
# ==============================================================================
with tab3:
    st.subheader("ℹ️ Kiến Trúc Hệ Thống & Cơ Sở Tri Thức Bệnh Cá")
    
    st.markdown("""
    ### 1. Kiến Trúc "Ontology-Driven Hybrid Reasoning with Python Execution Engine"
    Hệ thống được thiết kế theo nguyên tắc phân tầng chặt chẽ:
    1. **Tầng Thị giác Máy tính (Deep Learning):** Mô hình **EfficientNet-B0** phân tích ảnh chụp cá, trả về phân phối xác suất Top-3 lớp bệnh và kích hoạt thuật toán **Grad-CAM** trích xuất bản đồ nhiệt từ tầng tích chập cuối `features[-1]`.
    2. **Tầng Cơ sở Tri thức (Knowledge Base):** Bản thể học **FishDiseaseOntology.owl** (chuẩn W3C OWL 2 DL) chuẩn hóa phân cấp bệnh học thủy sản, tác nhân sinh học, danh mục triệu chứng, Datatype Facets ranh giới môi trường và các luật dịch tễ (`EnvironmentalRiskRule`, `ConflictResolutionRule`).
    3. **Tầng Suy luận Ngữ nghĩa (Semantic Execution Engine):** Module Python đóng vai trò động cơ thực thi: sử dụng **SPARQL queries** đọc động các luật và ngưỡng từ file OWL, tính toán Điểm Hỗ trợ Hỗn hợp (**Hybrid Support Score**) và phát hiện mâu thuẫn logic khi AI dự đoán sai lệch với triệu chứng/môi trường.
    4. **Tầng Lưu trữ & Kiểm toán (Persistence & Audit Trail):** Toàn bộ dữ liệu ca bệnh, ảnh, Grad-CAM, kết quả AI, các rule kích hoạt và phác đồ điều trị được lưu trữ trong CSDL quan hệ SQLite (`Database/FishDiagnosis.db`).
    """)
    
    st.markdown("### 2. Bảng Tra Cứu 4 Nhóm Trạng Thái Bệnh Cá & Ngưỡng Dịch Tễ")
    st.table([
        {
            "Lớp bệnh": "Bacterial Gill Diseases (BGD)",
            "Tên tiếng Việt": "Bệnh thối mang do vi khuẩn",
            "Tác nhân gây bệnh": "Flavobacterium branchiophilum (Vi khuẩn sợi)",
            "Triệu chứng điển hình": "Mang xơ xác, hoại tử tơ mang, dính nhớt",
            "Ngưỡng kích phát dịch tễ": "DO < 3.5 mg/L ; Ammonia > 0.5 mg/L"
        },
        {
            "Lớp bệnh": "Bacterial Red Diseases (BRD)",
            "Tên tiếng Việt": "Bệnh đốm đỏ / Nhiễm trùng huyết",
            "Tác nhân gây bệnh": "Aeromonas hydrophila (Trực khuẩn gram âm)",
            "Triệu chứng điển hình": "Loét da, xuất huyết thân/vây, chướng bụng",
            "Ngưỡng kích phát dịch tễ": "Ammonia > 0.8 mg/L ; Nhiệt độ thay đổi đột ngột"
        },
        {
            "Lớp bệnh": "Epizootic Ulcerative Syndrome (EUS)",
            "Tên tiếng Việt": "Hội chứng lở loét dịch tễ (Bệnh nấm)",
            "Tác nhân gây bệnh": "Aphanomyces invadans (Nấm thủy mi Oomycete)",
            "Triệu chứng điển hình": "Vết loét sâu ăn vào cơ thịt, u hạt viêm",
            "Ngưỡng kích phát dịch tễ": "Nhiệt độ < 25.0°C (Mùa lạnh) ; pH < 6.5"
        },
        {
            "Lớp bệnh": "Healthy Fish",
            "Tên tiếng Việt": "Cá khỏe mạnh",
            "Tác nhân gây bệnh": "Không có",
            "Triệu chứng điển hình": "Ngoại hình bình thường, phản xạ nhanh",
            "Ngưỡng kích phát dịch tễ": "Các chỉ số môi trường nằm trong ngưỡng an toàn"
        }
    ])
