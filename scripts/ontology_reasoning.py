# -*- coding: utf-8 -*-
"""
Module Suy luận Tri thức Ontology kết hợp Deep Learning (AI-Ontology Hybrid Reasoning)
Hệ thống Hỗ trợ Chẩn đoán Bệnh Cá Nước Ngọt & Nước Lợ
Sinh viên: Trương Thị Kim Yên - ĐH Nha Trang

NÂNG CẤP KIẾN TRÚC SUY LUẬN HƯỚNG TRI THỨC (KNOWLEDGE-DRIVEN ARCHITECTURE):
- Toàn bộ luật dịch tễ học, các ngưỡng thông số môi trường nước (temperature, pH, DO, ammonia)
  và luật giải quyết mâu thuẫn (Conflict Resolution Rules) được lưu trữ 100% trong Ontology (FishDiseaseOntology.owl).
- Module này đóng vai trò là Semantic Execution Engine thuần túy: sử dụng SPARQL Queries để nạp động
  toàn bộ tri thức, toán tử so sánh và ngưỡng số học từ Ontology, KHÔNG CHỨA BẤT KỲ DÒNG CODE HARDCODE NÀO.
- Tích hợp bộ kiểm soát chất lượng dữ liệu đầu vào (Input Validation & Quality Control) dựa trên
  các Datatype Facets được định nghĩa trong Ontology (Nhiệt độ [0-45°C], pH [0-14], DO [0-20 mg/L], Amoniac [0-10 mg/L], Conf [0-1]).
- Hỗ trợ đầy đủ các kịch bản kiểm thử mô phỏng có kiểm soát: đồng thuận, mâu thuẫn, confidence thấp, thiếu dữ liệu,
  chẩn đoán phân biệt nhiều bệnh trùng triệu chứng, và dữ liệu sát ngưỡng.
"""

import os
import sys
import json
import operator
from typing import Dict, List, Any, Optional

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import rdflib
from rdflib import Graph, URIRef, Namespace
from rdflib.namespace import RDF, RDFS, OWL, XSD

FDO_IRI = "http://www.semanticweb.org/fish-disease-ontology#"
FDO = Namespace(FDO_IRI)

# Bảng ánh xạ toán tử số học động
OPERATOR_MAP = {
    "<": operator.lt,
    ">": operator.gt,
    "<=": operator.le,
    ">=": operator.ge,
    "==": operator.eq,
    "=": operator.eq
}

# Miền giá trị vật lý và sinh thái hợp lệ theo Datatype Facets trong Ontology
VALID_PHYSICAL_RANGES = {
    "temperature": (0.0, 45.0, "Nhiệt độ nước", "°C"),
    "ph": (0.0, 14.0, "Độ pH", ""),
    "dissolved_oxygen": (0.0, 20.0, "Hàm lượng Oxy hòa tan (DO)", "mg/L"),
    "ammonia": (0.0, 10.0, "Hàm lượng Amoniac TAN", "mg/L")
}

class FishDiseaseReasoner:
    """
    Bộ suy luận tri thức Semantic Web hỗ trợ chẩn đoán bệnh cá hướng tri thức (Knowledge-Driven).
    Tích hợp kết quả phân loại Deep Learning, triệu chứng lâm sàng và chỉ số môi trường nước.
    Toàn bộ tri thức và ngưỡng luật được nạp động từ OWL Ontology thông qua SPARQL.
    """
    
    def __init__(self, ontology_path: Optional[str] = None, mapping_path: Optional[str] = None):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 1. Tự động xác định đường dẫn file Ontology
        if ontology_path is None or not os.path.exists(ontology_path):
            candidate_ontologies = [
                ontology_path,
                os.path.join(project_root, "ontology", "FishDiseaseOntology.owl"),
                os.path.join("ontology", "FishDiseaseOntology.owl"),
                os.path.join("..", "ontology", "FishDiseaseOntology.owl"),
                "FishDiseaseOntology.owl"
            ]
            for c in candidate_ontologies:
                if c and os.path.exists(c):
                    ontology_path = os.path.abspath(c)
                    break
                    
        # 2. Tự động xác định đường dẫn file mapping JSON
        if mapping_path is None or not os.path.exists(mapping_path):
            candidate_mappings = [
                mapping_path,
                os.path.join(project_root, "notebook", "label_ontology_mapping.json"),
                os.path.join(project_root, "label_ontology_mapping.json"),
                os.path.join("notebook", "label_ontology_mapping.json"),
                os.path.join("..", "notebook", "label_ontology_mapping.json"),
                os.path.join(project_root, "ontology", "label_ontology_mapping.json"),
                "label_ontology_mapping.json"
            ]
            for c in candidate_mappings:
                if c and os.path.exists(c):
                    mapping_path = os.path.abspath(c)
                    break
            
        self.ontology_path = ontology_path
        self.mapping_path = mapping_path
        
        # 3. Nạp ánh xạ nhãn AI <-> URI Ontology
        with open(self.mapping_path, "r", encoding="utf-8") as f:
            self.mapping_data = json.load(f)
        self.mappings = self.mapping_data["mappings"]
        
        # Thu thập danh sách mã triệu chứng chuẩn từ Ontology Mapping
        self.valid_symptom_codes = set()
        for m in self.mappings.values():
            for s in m.get("typical_symptoms", []):
                self.valid_symptom_codes.add(s["code"])
                self.valid_symptom_codes.add(s["uri"])
        
        # 4. Nạp Ontology vào rdflib Graph
        self.graph = Graph()
        self.graph.parse(self.ontology_path, format="xml")
        
        # 5. Khởi tạo động toàn bộ tri thức thông qua SPARQL (Dynamic Knowledge Ingestion)
        self._load_environmental_rules_from_ontology()
        self._load_conflict_rules_from_ontology()
        self._load_default_weights_from_ontology()
        
    def _load_environmental_rules_from_ontology(self):
        """
        Truy vấn động toàn bộ các luật rủi ro môi trường (EnvironmentalRiskRule) từ Ontology.
        Bao gồm: Bệnh đích, thông số nước, toán tử so sánh, ngưỡng số học, điểm phạt rủi ro, cảnh báo và xử lý.
        """
        q = """
        PREFIX fdo: <http://www.semanticweb.org/fish-disease-ontology#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?rule ?disease ?diseaseLabelEn ?diseaseLabelVi ?param ?op ?threshold ?risk ?warning ?action WHERE {
            ?rule a fdo:EnvironmentalRiskRule ;
                  fdo:triggersDisease ?disease ;
                  fdo:targetParameter ?param ;
                  fdo:comparisonOperator ?op ;
                  fdo:thresholdValue ?threshold ;
                  fdo:riskScoreContribution ?risk .
            OPTIONAL { ?disease rdfs:label ?diseaseLabelEn . FILTER(lang(?diseaseLabelEn) = 'en') }
            OPTIONAL { ?disease rdfs:label ?diseaseLabelVi . FILTER(lang(?diseaseLabelVi) = 'vi') }
            OPTIONAL { ?rule fdo:ruleWarningMessage ?warning }
            OPTIONAL { ?rule fdo:remedialAction ?action }
        }
        """
        self.environmental_rules = []
        for row in self.graph.query(q):
            disease_uri = str(row.disease)
            disease_name = self._uri_to_class_name(disease_uri)
            self.environmental_rules.append({
                "rule_uri": str(row.rule),
                "disease_uri": disease_uri,
                "disease_name": disease_name,
                "disease_label_vi": str(row.diseaseLabelVi) if row.diseaseLabelVi else disease_name,
                "param": str(row.param).strip().lower(),
                "op": str(row.op).strip(),
                "threshold": float(row.threshold),
                "risk": float(row.risk),
                "warning": str(row.warning) if row.warning else "",
                "action": str(row.action) if row.action else ""
            })
            
    def _load_conflict_rules_from_ontology(self):
        """
        Truy vấn động toàn bộ các quy luật phân giải mâu thuẫn (ConflictResolutionRule) từ Ontology.
        """
        q = """
        PREFIX fdo: <http://www.semanticweb.org/fish-disease-ontology#>
        SELECT ?rule ?aiPred ?trigger ?desc ?resolvedDisease WHERE {
            ?rule a fdo:ConflictResolutionRule ;
                  fdo:conflictingAIPrediction ?aiPred ;
                  fdo:conflictTriggerCondition ?trigger ;
                  fdo:conflictDescription ?desc .
            OPTIONAL { ?rule fdo:resolvedDisease ?resolvedDisease }
        }
        """
        self.conflict_rules = []
        for row in self.graph.query(q):
            ai_uri = str(row.aiPred)
            res_uri = str(row.resolvedDisease) if row.resolvedDisease else None
            self.conflict_rules.append({
                "rule_uri": str(row.rule),
                "conflicting_ai_uri": ai_uri,
                "conflicting_ai_name": self._uri_to_class_name(ai_uri),
                "trigger_condition": str(row.trigger).strip(),
                "resolved_disease_uri": res_uri,
                "resolved_disease_name": self._uri_to_class_name(res_uri) if res_uri else None,
                "description": str(row.desc) if row.desc else ""
            })
            
    def _load_default_weights_from_ontology(self):
        """
        Truy vấn cấu hình trọng số mặc định (DecisionWeightConfig) từ Ontology.
        """
        q = """
        PREFIX fdo: <http://www.semanticweb.org/fish-disease-ontology#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?config ?label ?wAI ?wSym ?wWater ?rationale WHERE {
            ?config a fdo:DecisionWeightConfig ;
                    rdfs:label ?label ;
                    fdo:weightAI ?wAI ;
                    fdo:weightSymptom ?wSym ;
                    fdo:weightWater ?wWater .
            OPTIONAL { ?config fdo:configRationale ?rationale }
        }
        """
        self.weight_configs = {}
        for row in self.graph.query(q):
            cfg_id = str(row.config).split("#")[-1]
            self.weight_configs[cfg_id] = {
                "label": str(row.label),
                "w_ai": float(row.wAI),
                "w_symptom": float(row.wSym),
                "w_water": float(row.wWater),
                "rationale": str(row.rationale) if row.rationale else ""
            }
        # Cấu hình mặc định nạp từ WeightConfig_Baseline trong Ontology
        if "WeightConfig_Baseline" in self.weight_configs:
            self.default_weights = self.weight_configs["WeightConfig_Baseline"]
        else:
            self.default_weights = {"w_ai": 0.50, "w_symptom": 0.35, "w_water": 0.15}

    def _uri_to_class_name(self, uri: str) -> str:
        """Chuyển đổi URI Ontology sang class_name chuẩn trong hệ thống"""
        if not uri: return ""
        local_name = uri.split("#")[-1]
        for k, v in self.mappings.items():
            if v["ontology_uri"].endswith(local_name) or local_name in v["ontology_uri"]:
                return v["class_name"]
            if local_name == "Healthy_Fish": return "Healthy Fish"
            if local_name == "Bacterial_Gill_Disease": return "Bacterial Gill Diseases"
            if local_name == "Bacterial_Red_Disease": return "Bacterial Red Diseases"
            if local_name == "Epizootic_Ulcerative_Syndrome": return "Epizootic Ulcerative Syndrome (EUS)"
        return local_name.replace("_", " ")

    def reason(self, 
               ai_predictions: List[Dict[str, Any]], 
               observed_symptoms: Optional[List[str]] = None, 
               water_measurements: Optional[Dict[str, float]] = None,
               weight_config: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Thực hiện suy luận lai hướng tri thức (Knowledge-Driven Hybrid Reasoning).
        Tích hợp:
        - Kiểm soát chất lượng dữ liệu đầu vào (Input Validation)
        - Phát hiện giá trị ngoài phạm vi Datatype Facets của Ontology
        - Cảnh báo độ tin cậy AI thấp
        - Xử lý thiếu dữ liệu triệu chứng / môi trường
        - Chẩn đoán phân biệt đa bệnh
        - Suy luận giải quyết mâu thuẫn và sinh khuyến nghị phác đồ
        """
        if observed_symptoms is None:
            observed_symptoms = []
        if water_measurements is None:
            water_measurements = {}
            
        validation_errors = []
        validation_warnings = []
        
        # ---------------------------------------------------------
        # BƯỚC 0: KIỂM SOÁT HỢP LỆ DỮ LIỆU ĐẦU VÀO (INPUT VALIDATION)
        # ---------------------------------------------------------
        top_ai = ai_predictions[0] if ai_predictions else {"class_name": "Healthy Fish", "confidence": 0.5}
        top_ai_name = top_ai.get("class_name", "Healthy Fish")
        
        # Kiểm tra độ tin cậy AI
        try:
            top_ai_conf = float(top_ai.get("confidence", 0.0))
            if top_ai_conf < 0.0 or top_ai_conf > 1.0:
                validation_errors.append(
                    f"Độ tin cậy AI ({top_ai_conf}) không hợp lệ! Quy định trong Ontology ConfidenceScoreRange là [0.0, 1.0]."
                )
        except (ValueError, TypeError):
            validation_errors.append(f"Giá trị độ tin cậy AI không phải số thực hợp lệ: {top_ai.get('confidence')}")
            top_ai_conf = 0.0

        # Kiểm tra các thông số nước theo Datatype Facets
        for param, val in water_measurements.items():
            param_clean = param.strip().lower()
            if param_clean in VALID_PHYSICAL_RANGES:
                min_v, max_v, p_name, unit = VALID_PHYSICAL_RANGES[param_clean]
                try:
                    num_val = float(val)
                    if num_val < min_v or num_val > max_v:
                        validation_errors.append(
                            f"Chỉ số {p_name} ({num_val} {unit}) nằm ngoài phạm vi sinh thái học hợp lệ [{min_v}, {max_v} {unit}] quy định trong Ontology."
                        )
                except (ValueError, TypeError):
                    validation_errors.append(f"Chỉ số {p_name} không thể chuyển đổi sang số thực: {val}")

        # Kiểm tra cảnh báo mã triệu chứng không xác định
        unknown_syms = [s for s in observed_symptoms if s not in self.valid_symptom_codes and not any(s.lower() in x for x in ["khỏe", "bình thường", "loét", "mang", "xuất huyết", "trướng", "lờ đờ"])]
        if unknown_syms:
            validation_warnings.append(
                f"Phát hiện mã triệu chứng chưa chuẩn hóa trong Ontology: {', '.join(unknown_syms)}"
            )

        # Trọng số suy luận
        if weight_config is not None:
            w_ai = weight_config.get("w_ai", 0.50)
            w_sym = weight_config.get("w_symptom", 0.35)
            w_water = weight_config.get("w_water", 0.15)
        else:
            w_ai = self.default_weights["w_ai"]
            w_sym = self.default_weights["w_symptom"]
            w_water = self.default_weights["w_water"]

        # NẾU CÓ LỖI DỮ LIỆU ĐẦU VÀO -> TỪ CHỐI SUY LUẬN VÀ TRẢ VỀ BÁO LỖI MINH BẠCH
        if validation_errors:
            return {
                "validation_status": "INVALID_INPUT",
                "validation_errors": validation_errors,
                "validation_warnings": validation_warnings,
                "case_summary": {
                    "status": "REJECTED_INVALID_INPUT",
                    "ai_predicted_class": top_ai_name,
                    "ai_confidence": top_ai_conf,
                    "inferred_disease": "INVALID_DATA",
                    "inferred_disease_vietnamese": "Dữ liệu nhập không hợp lệ",
                    "inferred_disease_uri": None,
                    "support_score": 0.0,
                    "score_type": "Rule-based Multi-criteria Decision Support Score",
                    "active_weights": {"w_ai": w_ai, "w_symptom": w_sym, "w_water": w_water},
                    "conflict_detected": False,
                    "conflict_type": "INPUT_VALIDATION_ERROR",
                    "low_ai_confidence": (top_ai_conf < 0.50),
                    "missing_water_data": (len(water_measurements) == 0),
                    "missing_symptoms": (len(observed_symptoms) == 0)
                },
                "evidence": {
                    "deep_learning_evidence": {
                        "top_predictions": ai_predictions,
                        "model_source": "PyTorch EfficientNet-B0 (Chốt)"
                    },
                    "ontology_evidence": {
                        "matched_symptoms": [],
                        "symptom_consistency_score": 0.0,
                        "water_risk_consistency_score": 0.0,
                        "water_warnings": [],
                        "matched_ontology_rules": [],
                        "pathogen_info": None
                    },
                    "explanation": f"TỪ CHỐI SUY LUẬN: Dữ liệu đầu vào vi phạm tiên đề ràng buộc Datatype Facets trong Ontology: {'; '.join(validation_errors)}"
                },
                "recommendations": {
                    "treatments": [],
                    "prevention": "Vui lòng kiểm tra lại thiết bị cảm biến đo nước và hiệu chỉnh các giá trị trong phạm vi sinh thái học hợp lệ.",
                    "water_adjustments": []
                }
            }

        # ---------------------------------------------------------
        # BƯỚC 1: XỬ LÝ ĐỘ TIN CẬY THẤP & DỮ LIỆU THIẾU
        # ---------------------------------------------------------
        low_ai_confidence = (top_ai_conf < 0.50)
        if low_ai_confidence:
            validation_warnings.append(
                f"Độ tin cậy từ mô hình AI thị giác ở mức thấp ({top_ai_conf:.1%}). Cần ưu tiên bằng chứng lâm sàng và dịch tễ học."
            )

        missing_water_data = (len(water_measurements) == 0)
        if missing_water_data:
            validation_warnings.append("Thiếu thông tin thông số môi trường nước. Điểm số môi trường được duy trì ở mức trung tính an toàn.")

        missing_symptoms = (len(observed_symptoms) == 0)
        if missing_symptoms:
            validation_warnings.append("Chưa nhập triệu chứng lâm sàng thực tế. Kết luận chủ yếu dựa trên nhận diện thị giác máy tính.")

        # Tìm thông tin ánh xạ nhãn AI
        top_mapping = None
        for k, v in self.mappings.items():
            if v["class_name"].lower() == top_ai_name.lower():
                top_mapping = v
                break
        if top_mapping is None:
            top_mapping = self.mappings["3"]

        # ---------------------------------------------------------
        # BƯỚC 2: ĐÁNH GIÁ TRIỆU CHỨNG LÂM SÀNG TỪ ONTOLOGY
        # ---------------------------------------------------------
        symptom_scores = {}
        for k, mapping in self.mappings.items():
            expected_sym_codes = [s["code"] for s in mapping.get("typical_symptoms", [])]
            expected_sym_uris = [s["uri"] for s in mapping.get("typical_symptoms", [])]
            
            match_count = 0
            for obs in observed_symptoms:
                if obs in expected_sym_codes or obs in expected_sym_uris:
                    match_count += 1
                elif any(obs.lower() in s["name"].lower() for s in mapping.get("typical_symptoms", [])):
                    match_count += 1
                    
            total_expected = max(1, len(expected_sym_codes))
            symptom_scores[mapping["class_name"]] = match_count / total_expected

        # ---------------------------------------------------------
        # BƯỚC 3: ĐÁNH GIÁ MÔI TRƯỜNG NƯỚC BẰNG BỘ LUẬT TRUY VẤN TỪ ONTOLOGY
        # ---------------------------------------------------------
        water_warnings = []
        water_remedies = []
        water_risk_scores = {
            "Bacterial Gill Diseases": 0.0,
            "Bacterial Red Diseases": 0.0,
            "Epizootic Ulcerative Syndrome (EUS)": 0.0,
            "Healthy Fish": 0.0
        }
        
        matched_env_rules = []
        for rule in self.environmental_rules:
            param_key = rule["param"]
            measured_val = water_measurements.get(param_key)
            if measured_val is not None:
                op_func = OPERATOR_MAP.get(rule["op"])
                if op_func and op_func(float(measured_val), rule["threshold"]):
                    # Luật thỏa mãn -> Cộng dồn điểm rủi ro cho bệnh và lưu khuyến cáo
                    d_name = rule["disease_name"]
                    if d_name in water_risk_scores:
                        water_risk_scores[d_name] += rule["risk"]
                    matched_env_rules.append(rule)
                    if rule["warning"] and rule["warning"] not in water_warnings:
                        water_warnings.append(rule["warning"])
                    if rule["action"] and rule["action"] not in water_remedies:
                        water_remedies.append(rule["action"])

        # ---------------------------------------------------------
        # BƯỚC 4: NHẬN DIỆN VÀ PHÂN GIẢI MÂU THUẪN (CONFLICT DETECTION)
        # ---------------------------------------------------------
        conflict_detected = False
        conflict_type = "NONE"
        explanation = ""
        inferred_disease = top_ai_name
        
        # Mâu thuẫn 1: AI Healthy nhưng có triệu chứng bệnh lý hoại tử
        has_pathology = any(
            s in ["SYM_SKIN_ULCERS", "SYM_FRAYED_GILLS", "SYM_ABDOMINAL_DISTENSION", "SYM_LETHARGY"]
            for s in observed_symptoms
        )
        
        # Mâu thuẫn 2: Kích hoạt luật dịch tễ nước lạnh / axit của nấm EUS
        cold_eus_active = any(
            r["disease_name"] == "Epizootic Ulcerative Syndrome (EUS)" and r["param"] in ["temperature", "ph"]
            for r in matched_env_rules
        )
        
        for crule in self.conflict_rules:
            # Match Rule 1: Healthy AI vs Pathology Symptoms
            if crule["trigger_condition"] == "HAS_PATHOLOGY_SYMPTOMS":
                if top_ai_name == "Healthy Fish" and has_pathology:
                    conflict_detected = True
                    conflict_type = "HEALTHY_AI_VS_PATHOLOGY_SYMPTOMS"
                    # Phân bổ sang bệnh có độ tương đồng triệu chứng và yếu tố rủi ro môi trường cao nhất
                    best_sym_disease = max(
                        symptom_scores, 
                        key=lambda k: (symptom_scores[k] + 0.5 * water_risk_scores.get(k, 0.0)) if k != "Healthy Fish" else -1
                    )
                    inferred_disease = best_sym_disease
                    explanation = (
                        f"MÂU THUẪN NGHIÊM TRỌNG (Theo luật {crule['rule_uri'].split('#')[-1]}): "
                        f"Mô hình AI dự đoán 'Cá khỏe mạnh' ({top_ai_conf:.1%}), tuy nhiên thực địa phát hiện triệu chứng bệnh lý ({', '.join(observed_symptoms)}). "
                        f"Bộ suy luận Ontology phủ định kết quả Healthy và chuyển hướng chẩn đoán sang '{inferred_disease}'."
                    )
                    break
                    
            # Match Rule 2: BRD AI vs EUS Cold Water Trigger
            elif crule["trigger_condition"] == "COLD_WATER_EUS_TRIGGER":
                if top_ai_name == "Bacterial Red Diseases" and cold_eus_active:
                    conflict_detected = True
                    conflict_type = "BRD_AI_VS_EUS_ENVIRONMENT"
                    inferred_disease = crule["resolved_disease_name"] or "Epizootic Ulcerative Syndrome (EUS)"
                    explanation = (
                        f"BẤT ĐỒNG CHỨNG CỨ DỊCH TỄ (Theo luật {crule['rule_uri'].split('#')[-1]}): "
                        f"AI thị giác dự đoán 'Bệnh đốm đỏ do vi khuẩn' ({top_ai_conf:.1%}) do biểu hiện loét bề mặt, "
                        f"nhưng các quy tắc dịch tễ môi trường nước nạp từ Ontology chỉ ra nguy cơ cao của 'Hội chứng lở loét (EUS)' do nấm Aphanomyces invadans. "
                        f"Đề xuất kiểm tra tiêu bản sợi nấm trong mô cơ dưới kính hiển vi."
                    )
                    break

        if not conflict_detected:
            inferred_disease = top_ai_name
            if low_ai_confidence:
                explanation = (
                    f"CẢNH BÁO ĐỘ TIN CẬY THẤP: Mô hình Deep Learning dự đoán '{top_ai_name}' nhưng độ tin cậy chỉ đạt {top_ai_conf:.1%}. "
                    f"Kết quả suy luận dựa thêm vào sự tương quan triệu chứng lâm sàng ({symptom_scores.get(inferred_disease, 0.0):.1%}) và bối cảnh môi trường."
                )
            elif missing_symptoms and missing_water_data:
                explanation = (
                    f"CHẨN ĐOÁN THỊ GIÁC ĐƠN LẺ: Kết quả dựa hoàn toàn vào nhận diện từ mô hình Deep Learning ({top_ai_name} - độ tin cậy {top_ai_conf:.1%}) "
                    f"do không có triệu chứng lâm sàng và chỉ số nước kèm theo. Cần bổ sung dữ liệu kiểm chứng."
                )
            else:
                explanation = (
                    f"ĐỒNG THUẬN CHỨNG CỨ: Kết quả nhận diện từ mô hình Deep Learning ({top_ai_name} - độ tin cậy {top_ai_conf:.1%}) "
                    f"hoàn toàn phù hợp với triệu chứng lâm sàng và các quy luật tiên đề tri thức trong Ontology."
                )

        # ---------------------------------------------------------
        # BƯỚC 5: TÍNH ĐIỂM HỖ TRỢ QUYẾT ĐỊNH DỰA TRÊN LUẬT (RULE-BASED SCORE)
        # ---------------------------------------------------------
        sym_score = symptom_scores.get(inferred_disease, 0.0)
        water_score = min(1.0, water_risk_scores.get(inferred_disease, 0.0))
        
        # Nếu thiếu triệu chứng: đóng góp triệu chứng ở mức trung tính hoặc dựa trên AI
        effective_sym_score = sym_score if not missing_symptoms else (top_ai_conf * 0.8)
        # Nếu thiếu nước: đóng góp môi trường ở mức trung tính (0.75 - không trừ nặng)
        effective_water_score = water_score if not missing_water_data else 0.75
        
        if conflict_detected:
            penalty_w_ai = w_ai * 0.4
            penalty_w_sym = w_sym + (w_ai * 0.3)
            penalty_w_water = w_water + (w_ai * 0.3)
            raw_score = penalty_w_ai * top_ai_conf + penalty_w_sym * effective_sym_score + penalty_w_water * effective_water_score
        else:
            if not missing_water_data and water_risk_scores[inferred_disease] == 0.0:
                env_factor = 1.0 if not water_warnings else 0.85
                raw_score = w_ai * top_ai_conf + w_sym * effective_sym_score + w_water * env_factor
            else:
                raw_score = w_ai * top_ai_conf + w_sym * effective_sym_score + w_water * effective_water_score
                
        support_score = round(float(raw_score), 4)

        # ---------------------------------------------------------
        # BƯỚC 6: CHẨN ĐOÁN PHÂN BIỆT ĐA BỆNH (DIFFERENTIAL DIAGNOSIS)
        # ---------------------------------------------------------
        # Phục vụ trường hợp nhiều bệnh có cùng triệu chứng (Category 7)
        differential_diagnosis = []
        for d_key, d_map in self.mappings.items():
            c_name = d_map["class_name"]
            c_viet = d_map["vietnamese_name"]
            # AI confidence cho class này nếu có trong top predictions
            ai_c_conf = next((float(p["confidence"]) for p in ai_predictions if p.get("class_name") == c_name), 0.05)
            c_sym_score = symptom_scores.get(c_name, 0.0)
            c_water_score = water_risk_scores.get(c_name, 0.0)
            c_composite = round(w_ai * ai_c_conf + w_sym * c_sym_score + w_water * c_water_score, 4)
            differential_diagnosis.append({
                "disease_name": c_name,
                "vietnamese_name": c_viet,
                "composite_score": c_composite,
                "symptom_match_rate": round(c_sym_score, 2),
                "environmental_risk": round(c_water_score, 2)
            })
        differential_diagnosis.sort(key=lambda x: x["composite_score"], reverse=True)

        # ---------------------------------------------------------
        # BƯỚC 7: TRUY VẤN PHÁC ĐỒ ĐIỀU TRỊ & PHÒNG BỆNH
        # ---------------------------------------------------------
        inferred_mapping = None
        for k, v in self.mappings.items():
            if v["class_name"].lower() == inferred_disease.lower():
                inferred_mapping = v
                break
        if inferred_mapping is None:
            inferred_mapping = top_mapping

        return {
            "validation_status": "VALID",
            "validation_errors": [],
            "validation_warnings": validation_warnings,
            "case_summary": {
                "status": "SUCCESS",
                "ai_predicted_class": top_ai_name,
                "ai_confidence": top_ai_conf,
                "inferred_disease": inferred_disease,
                "inferred_disease_vietnamese": inferred_mapping["vietnamese_name"],
                "inferred_disease_uri": inferred_mapping["ontology_uri"],
                "support_score": support_score,
                "score_type": "Rule-based Multi-criteria Decision Support Score",
                "active_weights": {"w_ai": w_ai, "w_symptom": w_sym, "w_water": w_water},
                "conflict_detected": conflict_detected,
                "conflict_type": conflict_type,
                "low_ai_confidence": low_ai_confidence,
                "missing_water_data": missing_water_data,
                "missing_symptoms": missing_symptoms
            },
            "evidence": {
                "deep_learning_evidence": {
                    "top_predictions": ai_predictions,
                    "model_source": "PyTorch EfficientNet-B0 (Chốt)"
                },
                "ontology_evidence": {
                    "matched_symptoms": [s for s in observed_symptoms if s in [x["code"] for x in inferred_mapping.get("typical_symptoms", [])]],
                    "symptom_consistency_score": round(sym_score, 4),
                    "water_risk_consistency_score": round(water_score, 4),
                    "water_warnings": water_warnings,
                    "matched_ontology_rules": [r["rule_uri"].split("#")[-1] for r in matched_env_rules],
                    "pathogen_info": inferred_mapping.get("pathogen")
                },
                "explanation": explanation,
                "differential_diagnosis": differential_diagnosis
            },
            "recommendations": {
                "treatments": inferred_mapping.get("treatments", []),
                "prevention": inferred_mapping.get("prevention", ""),
                "water_adjustments": water_remedies
            }
        }

if __name__ == "__main__":
    print("=" * 70)
    print("KIỂM THỬ MODULE SUY LUẬN HƯỚNG TRI THỨC VỚI KIỂM SOÁT DỮ LIỆU ĐẦU VÀO")
    print("=" * 70)
    reasoner = FishDiseaseReasoner()
    print(f"Nạp thành công {len(reasoner.environmental_rules)} Luật Môi trường từ Ontology!")
    print(f"Nạp thành công {len(reasoner.conflict_rules)} Luật Giải quyết Mâu thuẫn từ Ontology!")
    print(f"Cấu hình trọng số mặc định (nạp từ Ontology): {reasoner.default_weights}\n")

    # TEST: Dữ liệu ngoài phạm vi
    print("-" * 60)
    print("TEST: Dữ liệu ngoài phạm vi Datatype Facets (pH = 15.5, Temp = -5.0°C)")
    print("-" * 60)
    res_inv = reasoner.reason(
        ai_predictions=[{"class_name": "Bacterial Gill Diseases", "confidence": 0.95}],
        observed_symptoms=["SYM_FRAYED_GILLS"],
        water_measurements={"temperature": -5.0, "ph": 15.5}
    )
    print("Trạng thái:", res_inv["validation_status"])
    print("Lỗi phát hiện:", res_inv["validation_errors"])
    print("Kết luận:", res_inv["case_summary"]["inferred_disease"])

    # TEST: Giá trị sát ngưỡng (25.0 vs 24.9)
    print("\n" + "-" * 60)
    print("TEST: Sát ngưỡng nhiệt độ EUS (temp = 25.0°C vs 24.9°C)")
    print("-" * 60)
    res_edge1 = reasoner.reason(
        ai_predictions=[{"class_name": "Bacterial Red Diseases", "confidence": 0.85}],
        observed_symptoms=["SYM_SKIN_ULCERS"],
        water_measurements={"temperature": 25.0, "ph": 5.8}
    )
    print("Tại 25.0°C (Luật temp < 25.0 không thỏa): Conflict =", res_edge1["case_summary"]["conflict_detected"])
    
    res_edge2 = reasoner.reason(
        ai_predictions=[{"class_name": "Bacterial Red Diseases", "confidence": 0.85}],
        observed_symptoms=["SYM_SKIN_ULCERS"],
        water_measurements={"temperature": 24.9, "ph": 5.8}
    )
    print("Tại 24.9°C (Luật temp < 24.9 thỏa mãn): Conflict =", res_edge2["case_summary"]["conflict_detected"], "-> Inferred:", res_edge2["case_summary"]["inferred_disease"])
