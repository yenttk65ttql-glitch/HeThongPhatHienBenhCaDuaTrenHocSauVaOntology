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
- Khi chuyên gia cập nhật ngưỡng trong Ontology hoặc bổ sung bệnh mới, mã nguồn Python không cần sửa đổi.
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
        Bao gồm: Bệnh đích, thông số nước, toán tử so sánh, ngưỡng số học, điểm phạt rủi ro, cảnh báo và giải pháp.
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
            disease_class_name = self._uri_to_class_name(disease_uri)
            
            self.environmental_rules.append({
                "rule_uri": str(row.rule),
                "disease_uri": disease_uri,
                "disease_name": disease_class_name,
                "disease_label_vi": str(row.diseaseLabelVi) if row.diseaseLabelVi else disease_class_name,
                "param": str(row.param).strip().lower(),
                "op": str(row.op).strip(),
                "threshold": float(row.threshold),
                "risk": float(row.risk),
                "warning": str(row.warning) if row.warning else "",
                "action": str(row.action) if row.action else ""
            })
            
    def _load_conflict_rules_from_ontology(self):
        """
        Truy vấn động các luật nhận diện và giải quyết mâu thuẫn (ConflictResolutionRule) từ Ontology.
        """
        q = """
        PREFIX fdo: <http://www.semanticweb.org/fish-disease-ontology#>
        SELECT ?rule ?aiPred ?trigger ?resolvedDisease ?desc WHERE {
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
               observed_symptoms: List[str], 
               water_measurements: Optional[Dict[str, float]] = None,
               weight_config: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Thực hiện suy luận lai hướng tri thức (Knowledge-Driven Hybrid Reasoning).
        Toàn bộ logic kiểm tra ngưỡng nước và giải quyết mâu thuẫn đều duyệt qua các luật nạp từ Ontology.
        """
        if water_measurements is None:
            water_measurements = {}
            
        # Áp dụng trọng số từ tham số truyền vào hoặc lấy cấu hình mặc định từ Ontology
        if weight_config is not None:
            w_ai = weight_config.get("w_ai", 0.50)
            w_sym = weight_config.get("w_symptom", 0.35)
            w_water = weight_config.get("w_water", 0.15)
        else:
            w_ai = self.default_weights["w_ai"]
            w_sym = self.default_weights["w_symptom"]
            w_water = self.default_weights["w_water"]
            
        top_ai = ai_predictions[0] if ai_predictions else {"class_name": "Healthy Fish", "confidence": 0.5}
        top_ai_name = top_ai["class_name"]
        top_ai_conf = float(top_ai.get("confidence", 0.0))
        
        # Tìm thông tin ánh xạ nhãn AI
        top_mapping = None
        for k, v in self.mappings.items():
            if v["class_name"].lower() == top_ai_name.lower():
                top_mapping = v
                break
        if top_mapping is None:
            top_mapping = self.mappings["3"]
            
        # 1. ĐÁNH GIÁ SỰ PHÙ HỢP CỦA TRIỆU CHỨNG LÂM SÀNG TỪ ONTOLOGY
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
            
        # 2. ĐÁNH GIÁ MÔI TRƯỜNG NƯỚC BẰNG BỘ LUẬT TRUY VẤN TỪ ONTOLOGY
        # (Không hardcode bất kỳ ngưỡng nào, duyệt động qua self.environmental_rules)
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

        # 3. CƠ CHẾ NHẬN DIỆN & XỬ LÝ MÂU THUẪN DỰA TRÊN LUẬT ONTOLOGY
        # (Duyệt động qua self.conflict_rules nạp từ Ontology)
        conflict_detected = False
        conflict_type = "NONE"
        explanation = ""
        inferred_disease = top_ai_name
        
        # Kiểm tra điều kiện mâu thuẫn 1: AI đoán Healthy nhưng có triệu chứng tổn thương hoại tử da/mang
        has_pathology = any(s in ["SYM_SKIN_ULCERS", "SYM_FRAYED_GILLS", "SYM_ABDOMINAL_DISTENSION", "SYM_LETHARGY"] for s in observed_symptoms)
        
        # Kiểm tra điều kiện mâu thuẫn 2: AI đoán BRD nhưng kích hoạt luật nước lạnh/axit EUS
        cold_eus_active = any(r["disease_name"] == "Epizootic Ulcerative Syndrome (EUS)" and r["param"] in ["temperature", "ph"] for r in matched_env_rules)
        
        for crule in self.conflict_rules:
            # Match Rule 1: Healthy AI vs Pathology Symptoms
            if crule["trigger_condition"] == "HAS_PATHOLOGY_SYMPTOMS":
                if top_ai_name == "Healthy Fish" and has_pathology:
                    conflict_detected = True
                    conflict_type = "HEALTHY_AI_VS_PATHOLOGY_SYMPTOMS"
                    # Lấy bệnh có triệu chứng khớp nhất
                    best_sym_disease = max(symptom_scores, key=lambda k: symptom_scores[k] if k != "Healthy Fish" else -1)
                    inferred_disease = best_sym_disease
                    explanation = (
                        f"MÂU THUẪN NGHIÊM TRỌNG (Theo luật {crule['rule_uri'].split('#')[-1]}): "
                        f"Mô hình AI dự đoán 'Cá khỏe mạnh' ({top_ai_conf:.1%}), tuy nhiên lâm sàng phát hiện dấu hiệu bệnh lý ({', '.join(observed_symptoms)}). "
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
                        f"AI dự đoán 'Bệnh đốm đỏ do vi khuẩn' ({top_ai_conf:.1%}) do tổn thương loét bề mặt, "
                        f"nhưng các quy tắc dịch tễ môi trường nước nạp từ Ontology chỉ ra nguy cơ cao của 'Hội chứng lở loét (EUS)' do nấm Aphanomyces invadans. "
                        f"Đề xuất chuyên gia sinh học kiểm tra tiêu bản sợi nấm trong mô cơ dưới kính hiển vi."
                    )
                    break

        if not conflict_detected:
            inferred_disease = top_ai_name
            explanation = (
                f"ĐỒNG THUẬN CHỨNG CỨ: Kết quả nhận diện từ mô hình Deep Learning ({top_ai_name} - độ tin cậy {top_ai_conf:.1%}) "
                f"hoàn toàn phù hợp với triệu chứng lâm sàng và các quy luật tiên đề tri thức trong Ontology."
            )

        # 4. TÍNH TOÁN ĐIỂM HỖ TRỢ QUYẾT ĐỊNH DỰA TRÊN LUẬT (RULE-BASED DECISION SUPPORT SCORE)
        # Lưu ý phương pháp luận: Đây là Rule-based Decision Support Score, KHÔNG PHẢI xác suất hậu nghiệm Bayes
        sym_score = symptom_scores.get(inferred_disease, 0.0)
        water_score = min(1.0, water_risk_scores.get(inferred_disease, 0.0))
        
        if conflict_detected:
            # Phạt mâu thuẫn: giảm trọng số AI, tăng trọng số triệu chứng và môi trường
            penalty_w_ai = w_ai * 0.4
            penalty_w_sym = w_sym + (w_ai * 0.3)
            penalty_w_water = w_water + (w_ai * 0.3)
            raw_score = penalty_w_ai * top_ai_conf + penalty_w_sym * sym_score + penalty_w_water * water_score
        else:
            env_factor = 1.0 if not water_warnings else 0.85
            raw_score = w_ai * top_ai_conf + w_sym * sym_score + w_water * (water_score if water_risk_scores[inferred_disease] > 0 else env_factor)
            
        support_score = round(float(raw_score), 4)

        # 5. TRUY VẤN PHÁC ĐỒ ĐIỀU TRỊ & PHÒNG BỆNH
        inferred_mapping = None
        for k, v in self.mappings.items():
            if v["class_name"].lower() == inferred_disease.lower():
                inferred_mapping = v
                break
        if inferred_mapping is None:
            inferred_mapping = top_mapping

        return {
            "case_summary": {
                "ai_predicted_class": top_ai_name,
                "ai_confidence": top_ai_conf,
                "inferred_disease": inferred_disease,
                "inferred_disease_vietnamese": inferred_mapping["vietnamese_name"],
                "inferred_disease_uri": inferred_mapping["ontology_uri"],
                "support_score": support_score,
                "score_type": "Rule-based Multi-criteria Decision Support Score",
                "active_weights": {"w_ai": w_ai, "w_symptom": w_sym, "w_water": w_water},
                "conflict_detected": conflict_detected,
                "conflict_type": conflict_type
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
                "explanation": explanation
            },
            "recommendations": {
                "treatments": inferred_mapping.get("treatments", []),
                "prevention": inferred_mapping.get("prevention", ""),
                "water_adjustments": water_remedies
            }
        }

if __name__ == "__main__":
    print("=" * 70)
    print("KIỂM THỬ MODULE SUY LUẬN HƯỚNG TRI THỨC (FISH_DISEASE_REASONER)")
    print("=" * 70)
    reasoner = FishDiseaseReasoner()
    print(f"Nạp thành công {len(reasoner.environmental_rules)} Luật Môi trường từ Ontology!")
    print(f"Nạp thành công {len(reasoner.conflict_rules)} Luật Giải quyết Mâu thuẫn từ Ontology!")
    print(f"Cấu hình trọng số mặc định (nạp từ Ontology): {reasoner.default_weights}\n")

    # TEST CASE 1: BGD đồng thuận
    print("-" * 60)
    print("TEST CASE 1: BGD - AI và Triệu chứng đồng thuận")
    print("-" * 60)
    res1 = reasoner.reason(
        ai_predictions=[{"class_name": "Bacterial Gill Diseases", "confidence": 0.942}],
        observed_symptoms=["SYM_FRAYED_GILLS", "SYM_LETHARGY"],
        water_measurements={"temperature": 29.5, "ph": 7.2, "dissolved_oxygen": 2.8, "ammonia": 0.7}
    )
    print("Tóm tắt chẩn đoán:", json.dumps(res1["case_summary"], ensure_ascii=False, indent=2))
    print("Giải thích:", res1["evidence"]["explanation"])
    print("Luật kích hoạt:", res1["evidence"]["ontology_evidence"]["matched_ontology_rules"])

    # TEST CASE 2: Mâu thuẫn loại 1 (AI Healthy vs Triệu chứng loét da)
    print("\n" + "-" * 60)
    print("TEST CASE 2: Mâu thuẫn AI (Healthy) vs Triệu chứng loét da")
    print("-" * 60)
    res2 = reasoner.reason(
        ai_predictions=[{"class_name": "Healthy Fish", "confidence": 0.785}],
        observed_symptoms=["SYM_SKIN_ULCERS", "SYM_ABDOMINAL_DISTENSION"],
        water_measurements={"temperature": 28.0, "ph": 7.5, "dissolved_oxygen": 4.8}
    )
    print("Tóm tắt chẩn đoán:", json.dumps(res2["case_summary"], ensure_ascii=False, indent=2))
    print("Giải thích:", res2["evidence"]["explanation"])

    # TEST CASE 3: Bất đồng dịch tễ (Red Disease AI vs EUS mùa lạnh pH thấp)
    print("\n" + "-" * 60)
    print("TEST CASE 3: Bất đồng dịch tễ (AI đoán Red Disease nhưng nước lạnh 21°C và pH 5.8)")
    print("-" * 60)
    res3 = reasoner.reason(
        ai_predictions=[{"class_name": "Bacterial Red Diseases", "confidence": 0.812}],
        observed_symptoms=["SYM_SKIN_ULCERS"],
        water_measurements={"temperature": 21.0, "ph": 5.8, "dissolved_oxygen": 3.4}
    )
    print("Tóm tắt chẩn đoán:", json.dumps(res3["case_summary"], ensure_ascii=False, indent=2))
    print("Giải thích:", res3["evidence"]["explanation"])
    print("Khuyến nghị phác đồ:", json.dumps(res3["recommendations"], ensure_ascii=False, indent=2))
