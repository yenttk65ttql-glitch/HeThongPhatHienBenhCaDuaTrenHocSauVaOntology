# -*- coding: utf-8 -*-
import os
import sys

# Khắc phục xung đột thư viện OpenMP trên môi trường Windows (PyTorch + Matplotlib/OpenCV)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Union, Optional

# Đảm bảo đường dẫn gốc được nạp vào sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.predictor import FishDiseasePredictor
from src.gradcam import FishGradCAM
from scripts.ontology_reasoning import FishDiseaseReasoner

class FishDiseaseHybridPipeline:
    """
    Hệ thống Pipeline Tích Hợp Hoàn Chỉnh (End-to-End Hybrid Engine).
    Kết nối Mô hình Thị giác AI, Giải thích Grad-CAM và Suy luận Tri thức Ontology.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 class_to_idx_path: Optional[str] = None,
                 ontology_path: Optional[str] = None,
                 mapping_path: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Khởi tạo toàn bộ các thành phần của hệ thống.
        """
        print("Đang khởi tạo Pipeline Chẩn đoán Bệnh cá Lai (AI-Ontology)...")
        # 1. Khởi tạo bộ dự đoán thị giác Deep Learning
        self.predictor = FishDiseasePredictor(
            model_path=model_path,
            class_to_idx_path=class_to_idx_path,
            device=device
        )
        
        # 2. Khởi tạo bộ giải thích thị giác Grad-CAM
        self.grad_cam = FishGradCAM(
            model=self.predictor.model
        )
        
        # 3. Khởi tạo bộ suy luận tri thức Ontology (Knowledge-Driven)
        self.reasoner = FishDiseaseReasoner(
            ontology_path=ontology_path,
            mapping_path=mapping_path
        )
        print("-> Khởi tạo thành công toàn bộ Pipeline!")

    def diagnose(self, 
                 image_input: Union[str, Path, Any],
                 observed_symptoms: Optional[List[str]] = None,
                 water_measurements: Optional[Dict[str, float]] = None,
                 weight_config: Optional[Dict[str, float]] = None,
                 generate_gradcam: bool = True,
                 gradcam_save_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Thực thi quy trình chẩn đoán hoàn chỉnh từ ảnh cá bệnh thực tế.
        """
        pipeline_start_time = time.time()
        
        # =====================================================================
        # BƯỚC 1: DỰ ĐOÁN THỊ GIÁC AI (DEEP LEARNING PREDICTION)
        # =====================================================================
        pred_res = self.predictor.predict(image_input, top_k=3)
        top_predictions = pred_res["top_predictions"]
        top1_class = pred_res["top1_class"]
        top1_conf = pred_res["top1_confidence"]
        
        # =====================================================================
        # BƯỚC 2: TẠO BẢN ĐỒ NHIỆT GIẢI THÍCH GRAD-CAM
        # =====================================================================
        gradcam_info = {}
        if generate_gradcam:
            target_class_idx = top_predictions[0]["class_idx"]
            cam_mask, _, cam_conf = self.grad_cam.generate_cam(
                input_tensor=pred_res["tensor"],
                target_class=target_class_idx
            )
            
            saved_cam_path = None
            if gradcam_save_path:
                saved_cam_path = self.grad_cam.save_visualization(
                    pil_image=pred_res["pil_image"],
                    cam_mask=cam_mask,
                    output_path=gradcam_save_path,
                    predicted_class=top1_class,
                    confidence=top1_conf
                )
                
            gradcam_info = {
                "cam_generated": True,
                "target_class": top1_class,
                "saved_path": str(saved_cam_path) if saved_cam_path else None,
                "cam_mask_shape": cam_mask.shape
            }
        else:
            gradcam_info = {"cam_generated": False}
            
        # =====================================================================
        # BƯỚC 3: SUY LUẬN TRI THỨC VÀ PHÂN GIẢI MÂU THUẪN (ONTOLOGY REASONING)
        # =====================================================================
        # Chuyển đổi định dạng predictions để tương thích với FishDiseaseReasoner
        reasoner_ai_input = [
            {"class_name": p["class_name"], "confidence": p["confidence"]}
            for p in top_predictions
        ]
        
        reasoning_res = self.reasoner.reason(
            ai_predictions=reasoner_ai_input,
            observed_symptoms=observed_symptoms or [],
            water_measurements=water_measurements or {},
            weight_config=weight_config
        )
        
        total_time_ms = round((time.time() - pipeline_start_time) * 1000.0, 2)
        
        # =====================================================================
        # BƯỚC 4: TỔNG HỢP HỒ SƠ CHẨN ĐOÁN LÂM SÀNG TOÀN DIỆN
        # =====================================================================
        case_summary = reasoning_res["case_summary"]
        evidence = reasoning_res["evidence"]
        recommendations = reasoning_res["recommendations"]
        
        unified_result = {
            "metadata": {
                "pipeline_version": "2.0-Knowledge-Driven",
                "execution_time_total_ms": total_time_ms,
                "ai_inference_time_ms": pred_res["inference_time_ms"],
                "validation_status": reasoning_res.get("validation_status", "VALID"),
                "validation_errors": reasoning_res.get("validation_errors", []),
                "validation_warnings": reasoning_res.get("validation_warnings", [])
            },
            "deep_learning_results": {
                "model_name": "EfficientNet-B0 (Đã Khóa)",
                "top1_predicted_class": top1_class,
                "top1_confidence": top1_conf,
                "top3_predictions": top_predictions,
                "all_probabilities": pred_res["all_probabilities"],
                "gradcam": gradcam_info
            },
            "clinical_and_environmental_inputs": {
                "observed_symptoms": observed_symptoms or [],
                "water_measurements": water_measurements or {}
            },
            "ontology_hybrid_conclusion": {
                "inferred_disease": case_summary["inferred_disease"],
                "inferred_disease_vietnamese": case_summary["inferred_disease_vietnamese"],
                "inferred_disease_uri": case_summary["inferred_disease_uri"],
                "hybrid_support_score": case_summary["support_score"],
                "score_type": case_summary["score_type"],
                "conflict_detected": case_summary["conflict_detected"],
                "conflict_type": case_summary["conflict_type"],
                "explanation": evidence["explanation"],
                "matched_ontology_rules": evidence["ontology_evidence"]["matched_ontology_rules"],
                "differential_diagnosis": evidence.get("differential_diagnosis", [])
            },
            "actionable_recommendations": {
                "treatments": recommendations["treatments"],
                "prevention": recommendations["prevention"],
                "water_adjustments": recommendations["water_adjustments"]
            }
        }
        return unified_result

    def format_clinical_report(self, res: Dict[str, Any]) -> str:
        """
        Định dạng kết quả chẩn đoán thành bản báo cáo văn bản chuyên nghiệp.
        """
        meta = res["metadata"]
        dl = res["deep_learning_results"]
        inputs = res["clinical_and_environmental_inputs"]
        onto = res["ontology_hybrid_conclusion"]
        rec = res["actionable_recommendations"]
        
        lines = []
        lines.append("=" * 80)
        lines.append("HỆ THỐNG HỖ TRỢ CHẨN ĐOÁN BỆNH CÁ KẾT HỢP DEEP LEARNING & ONTOLOGY")
        lines.append("BÁO CÁO KẾT QUẢ CHẨN ĐOÁN LÂM SÀNG TOÀN DIỆN (CLINICAL DIAGNOSTIC REPORT)")
        lines.append("=" * 80)
        lines.append(f"Thời gian phân tích: {meta['execution_time_total_ms']} ms (AI Inference: {meta['ai_inference_time_ms']} ms)")
        lines.append(f"Trạng thái dữ liệu đầu vào: {meta['validation_status']}")
        if meta["validation_warnings"]:
            lines.append("Cảnh báo kiểm soát dữ liệu:")
            for w in meta["validation_warnings"]:
                lines.append(f"  [!] {w}")
        lines.append("-" * 80)
        
        lines.append("1. NHẬN DIỆN THỊ GIÁC MÁY TÍNH (DEEP LEARNING - EFFICIENTNET-B0):")
        for p in dl["top3_predictions"]:
            bar = "#" * int(p["confidence"] * 20)
            lines.append(f"  Top {p['rank']}: {p['class_name']:<35} | {p['confidence_percent']:>7} | {bar}")
        if dl["gradcam"].get("saved_path"):
            lines.append(f"  -> Bản đồ nhiệt Grad-CAM đã xuất: {dl['gradcam']['saved_path']}")
        lines.append("")
        
        lines.append("2. THÔNG TIN LÂM SÀNG & MÔI TRƯỜNG THỰC ĐỊA:")
        sym_str = ", ".join(inputs["observed_symptoms"]) if inputs["observed_symptoms"] else "Chưa ghi nhận triệu chứng"
        lines.append(f"  - Triệu chứng quan sát: {sym_str}")
        water_items = [f"{k}={v}" for k, v in inputs["water_measurements"].items()]
        water_str = ", ".join(water_items) if water_items else "Chưa cung cấp số liệu nước"
        lines.append(f"  - Chỉ số môi trường ao nuôi: {water_str}")
        lines.append("")
        
        lines.append("3. KẾT LUẬN SUY LUẬN TRI THỨC NGỮ NGHĨA (ONTOLOGY REASONING):")
        lines.append(f"  * BỆNH KẾT LUẬN: {onto['inferred_disease_vietnamese'].upper()} ({onto['inferred_disease']})")
        lines.append(f"  * ĐIỂM TIN CẬY LAI (Support Score): {onto['hybrid_support_score']} ({onto['score_type']})")
        lines.append(f"  * Phát hiện mâu thuẫn: {'CÓ - ' + onto['conflict_type'] if onto['conflict_detected'] else 'KHÔNG (Đồng thuận hoàn toàn)'}")
        lines.append(f"  * Giải thích minh bạch: {onto['explanation']}")
        if onto["matched_ontology_rules"]:
            lines.append(f"  * Quy luật sinh thái kích hoạt: {', '.join(onto['matched_ontology_rules'])}")
        lines.append("")
        
        if onto.get("differential_diagnosis"):
            lines.append("4. XẾP HẠNG CHẨN ĐOÁN PHÂN BIỆT (DIFFERENTIAL DIAGNOSIS):")
            for rank, d in enumerate(onto["differential_diagnosis"], 1):
                lines.append(f"  #{rank}: {d['vietnamese_name']:<30} | Điểm tổng hợp: {d['composite_score']} (Triệu chứng: {d['symptom_match_rate']*100:.0f}%, Rủi ro nước: {d['environmental_risk']*100:.0f}%)")
            lines.append("")
            
        lines.append("5. KHUYẾN NGHỊ PHÁC ĐỒ ĐIỀU TRỊ VÀ CẢI THIỆN MÔI TRƯỜNG (FAO/WOAH):")
        if rec["treatments"]:
            lines.append("  a) Phác đồ điều trị:")
            for t in rec["treatments"]:
                lines.append(f"     + {t}")
        if rec["water_adjustments"]:
            lines.append("  b) Xử lý nước cấp bách:")
            for w in rec["water_adjustments"]:
                lines.append(f"     + {w}")
        if rec["prevention"]:
            lines.append(f"  c) Phòng bệnh lâu dài: {rec['prevention']}")
            
        lines.append("=" * 80)
        return "\n".join(lines)

if __name__ == "__main__":
    print("=" * 80)
    print("KIỂM THỬ END-TO-END PIPELINE VỚI ẢNH THỰC TẾ (FISH_DISEASE_HYBRID_PIPELINE)")
    print("=" * 80)
    
    pipeline = FishDiseaseHybridPipeline()
    
    test_img = Path("data/cache_224/test.jpg")
    if not test_img.exists():
        # Tìm ảnh mẫu bất kỳ trong data/raw
        for r, d, fs in os.walk("data/raw"):
            for f in fs:
                if f.lower().endswith(".jpg"):
                    test_img = Path(r) / f
                    break
            if test_img.exists():
                break
                
    if test_img.exists():
        print(f"\nThực thi kiểm thử Pipeline hoàn chỉnh trên ảnh mẫu: {test_img}")
        
        # Mô phỏng tình huống 1: Đồng thuận BGD
        out_report = pipeline.diagnose(
            image_input=test_img,
            observed_symptoms=["SYM_FRAYED_GILLS", "SYM_LETHARGY"],
            water_measurements={"temperature": 29.0, "ph": 7.2, "dissolved_oxygen": 2.5, "ammonia": 0.7},
            generate_gradcam=True,
            gradcam_save_path="reports/demo_end_to_end_gradcam.png"
        )
        
        print("\n" + pipeline.format_clinical_report(out_report))
    else:
        print("Không tìm thấy tệp ảnh nào trong thư mục data.")
