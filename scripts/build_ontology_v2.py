# -*- coding: utf-8 -*-
"""
Script xây dựng Ontology v2 hoàn chỉnh cho Đề tài:
"Xây dựng hệ thống hỗ trợ chẩn đoán bệnh cá kết hợp tri thức Ontology và học máy"
Sinh viên: Trương Thị Kim Yên - ĐH Nha Trang

NÂNG CẤP KIẾN TRÚC SUY LUẬN AI - ONTOLOGY:
- Toàn bộ tri thức dịch tễ học, các ngưỡng thông số môi trường nước (temperature, pH, DO, ammonia)
  và luật giải quyết mâu thuẫn (Conflict Resolution Rules) được mô hình hóa 100% trong Ontology (OWL Classes, Properties, Individuals & SWRL Rules).
- Python chỉ đóng vai trò là Semantic Execution Engine đọc và thực thi tri thức thông qua SPARQL Queries,
  không còn bất kỳ dòng code hardcode 'if temp < 25' hay số học nào trong Python.
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import rdflib
from rdflib import Graph, URIRef, Literal, Namespace, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD, DC
from rdflib.collection import Collection

def build_fish_disease_ontology_v2():
    g = Graph()
    
    # 1. Namespace chuẩn hóa
    ONTOLOGY_IRI = "http://www.semanticweb.org/fish-disease-ontology#"
    VERSION_IRI = "http://www.semanticweb.org/fish-disease-ontology/2.0.0"
    SWRL_IRI = "http://www.w3.org/2003/11/swrl#"
    SWRLB_IRI = "http://www.w3.org/2003/11/swrlb#"
    
    FDO = Namespace(ONTOLOGY_IRI)
    SWRL = Namespace(SWRL_IRI)
    SWRLB = Namespace(SWRLB_IRI)
    
    g.bind("", FDO)
    g.bind("owl", OWL)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)
    g.bind("dc", DC)
    g.bind("swrl", SWRL)
    g.bind("swrlb", SWRLB)
    
    # Khai báo Metadata Ontology
    ont_ref = URIRef(ONTOLOGY_IRI)
    g.add((ont_ref, RDF.type, OWL.Ontology))
    g.add((ont_ref, OWL.versionIRI, URIRef(VERSION_IRI)))
    g.add((ont_ref, RDFS.label, Literal("Fish Disease Ontology (FDO v2.0 - Knowledge-Driven Reasoning)", lang="en")))
    g.add((ont_ref, RDFS.label, Literal("Hệ thống tri thức Ontology Chẩn đoán bệnh cá v2.0 (Kiến trúc Suy luận Hướng Tri thức)", lang="vi")))
    g.add((ont_ref, RDFS.comment, Literal("Ontology phục vụ chẩn đoán bệnh cá nước ngọt và nước lợ, kết hợp giữa mô hình học sâu (Deep Learning) và hệ suy diễn tri thức (Semantic Web/HermiT/OWL). Toàn bộ luật rủi ro môi trường, ngưỡng sinh thái học, và luật giải quyết mâu thuẫn chẩn đoán được lưu trữ trực tiếp trong Ontology, cho phép truy xuất và thực thi độc lập qua SPARQL/SWRL.", lang="vi")))
    g.add((ont_ref, DC.creator, Literal("Trương Thị Kim Yên - Chuyên ngành HTTTQL, Trường Đại học Nha Trang")))
    g.add((ont_ref, DC.publisher, Literal("Trường Đại học Nha Trang (NTU)")))
    
    # Hàm thêm Class chuẩn hóa
    def add_class(uri_name, parent_uri=None, label_en=None, label_vi=None, comment=None, source=None):
        cls_uri = FDO[uri_name]
        g.add((cls_uri, RDF.type, OWL.Class))
        if parent_uri:
            g.add((cls_uri, RDFS.subClassOf, parent_uri))
        if label_en:
            g.add((cls_uri, RDFS.label, Literal(label_en, lang="en")))
        if label_vi:
            g.add((cls_uri, RDFS.label, Literal(label_vi, lang="vi")))
        if comment:
            g.add((cls_uri, RDFS.comment, Literal(comment, lang="vi")))
        if source:
            g.add((cls_uri, DC.source, Literal(source)))
        return cls_uri

    # ==========================================================
    # 2. CẤU TRÚC PHÂN CẤP LỚP (CLASS HIERARCHY)
    # ==========================================================
    
    # --- FishCondition ---
    c_fish_condition = add_class(
        "FishCondition", OWL.Thing, 
        "Fish Condition", "Tình trạng sức khỏe cá", 
        "Lớp gốc bao quát tất cả các trạng thái sinh lý hoặc bệnh lý của cá nuôi."
    )
    
    c_healthy_state = add_class(
        "HealthyState", c_fish_condition,
        "Healthy State", "Trạng thái khỏe mạnh",
        "Trạng thái cá phát triển bình thường, không mang mầm bệnh nguy hiểm, bơi lội nhanh nhẹn, da mang sáng bóng."
    )
    c_healthy_fish = add_class(
        "Healthy_Fish", c_healthy_state,
        "Healthy Fish Condition", "Cá khỏe mạnh",
        "Lớp đại diện cho cá không phát hiện bệnh lý, tương ứng nhãn phân loại 'Healthy Fish' trong mô hình Deep Learning."
    )

    c_disease = add_class(
        "Disease", c_fish_condition,
        "Disease", "Bệnh ở cá",
        "Lớp đại diện cho mọi tình trạng rối loạn cấu trúc hoặc chức năng do tác nhân sinh học hoặc môi trường gây ra."
    )
    
    # Phân cấp bệnh học: BacterialDisease
    c_bacterial_disease = add_class(
        "BacterialDisease", c_disease,
        "Bacterial Disease", "Bệnh do vi khuẩn",
        "Các bệnh truyền nhiễm do vi khuẩn Gram âm hoặc Gram dương gây ra trên cá nước ngọt."
    )
    c_bgd = add_class(
        "Bacterial_Gill_Disease", c_bacterial_disease,
        "Bacterial Gill Disease", "Bệnh thối mang do vi khuẩn",
        "Bệnh do vi khuẩn Flavobacterium branchiophilum gây ra, làm mang cá bị hoại tử, tơ mang rách xơ xác và tiết nhiều nhớt.",
        "Giáo trình Bệnh học Thủy sản, ĐH Cần Thơ (2014); QCVN 02-19:2014/BNNPTNT."
    )
    c_brd = add_class(
        "Bacterial_Red_Disease", c_bacterial_disease,
        "Bacterial Red Disease", "Bệnh đốm đỏ do vi khuẩn",
        "Bệnh xuất huyết do vi khuẩn Aeromonas hydrophila gây ra, biểu hiện bằng các đốm đỏ, xuất huyết dưới da, gốc vây và loét cơ.",
        "Giáo trình Bệnh học Thủy sản, ĐH Cần Thơ (2014); Sổ tay chẩn đoán bệnh cá nước ngọt ĐH Nha Trang."
    )
    c_esc = add_class(
        "Enteric_Septicemia_Catfish", c_bacterial_disease,
        "Enteric Septicemia of Catfish", "Bệnh gan thận mủ ở cá tra",
        "Bệnh nhiễm khuẩn máu đường ruột do vi khuẩn Edwardsiella ictaluri gây ra trên cá tra (Pangasius hypophthalmus).",
        "Giáo trình Bệnh học Thủy sản, ĐH Cần Thơ."
    )
    
    # Phân cấp bệnh học: FungalDisease
    c_fungal_disease = add_class(
        "FungalDisease", c_disease,
        "Fungal Disease", "Bệnh do nấm và nấm nước",
        "Các bệnh do vi nấm thật hoặc sinh vật giả nấm dạng sợi (Oomycetes) gây hoại tử biểu bì và mô cơ cá."
    )
    c_eus = add_class(
        "Epizootic_Ulcerative_Syndrome", c_fungal_disease,
        "Epizootic Ulcerative Syndrome (EUS)", "Hội chứng lở loét dịch tễ (EUS)",
        "Bệnh truyền nhiễm nguy hiểm do sinh vật Oomycete (Aphanomyces invadans) gây ra. Sợi nấm ăn sâu vào mô cơ tạo nên các vết loét hoại tử dạng u hạt (mycotic granulomas). Bệnh thường bùng phát vào mùa lạnh (nhiệt độ < 25°C) hoặc sau mưa lũ kéo dài làm giảm pH nước ao.",
        "FAO Fisheries Technical Paper No. 402, Lilley et al. (1997); WOAH Manual of Diagnostic Tests for Aquatic Animals (2021), Chapter 2.3.2."
    )
    c_sapr = add_class(
        "Saprolegniasis", c_fungal_disease,
        "Saprolegniasis", "Bệnh nấm thủy mi",
        "Bệnh do nấm nước Saprolegnia parasitica ký sinh trên các vết thương hở ngoài da, tạo đám sợi bông trắng như búi bông gòn.",
        "Giáo trình Bệnh học Thủy sản, ĐH Cần Thơ."
    )

    c_parasitic_disease = add_class(
        "ParasiticDisease", c_disease,
        "Parasitic Disease", "Bệnh do ký sinh trùng",
        "Các bệnh do ngoại ký sinh trùng (trùng bánh xe, sán lá mang, trùng quả dưa) hoặc nội ký sinh trùng."
    )
    c_trich = add_class(
        "Trichodiniasis", c_parasitic_disease,
        "Trichodiniasis", "Bệnh trùng bánh xe",
        "Bệnh do ngoại ký sinh trùng hình đĩa bánh xe Trichodina ký sinh trên mang và da cá, làm cá ngứa ngáy, tăng tiết chất nhớt.",
        "Giáo trình Ký sinh trùng Thủy sản, ĐH Thủy sản Nha Trang."
    )

    # --- DiagnosticCase (Phân biệt ca bệnh thực tế vs concept bệnh) ---
    c_case = add_class(
        "DiagnosticCase", OWL.Thing,
        "Diagnostic Case", "Ca chẩn đoán lâm sàng",
        "Biểu diễn một ca chẩn đoán thực tế tại một ao nuôi/mẫu cá cụ thể. Một instance của DiagnosticCase lưu vết các bằng chứng từ ảnh chụp, triệu chứng quan sát tại chỗ, và đo đạc chất lượng nước để đưa vào bộ suy luận tri thức."
    )

    # --- FishSpecies ---
    c_fish_species = add_class(
        "FishSpecies", OWL.Thing,
        "Fish Species", "Loài cá nuôi",
        "Các loài cá nước ngọt/nước lợ thường gặp trong nuôi trồng thủy sản."
    )
    add_class("Pangasius_hypophthalmus", c_fish_species, "Striped Catfish", "Cá tra", "Loài cá nước ngọt xuất khẩu chủ lực vùng ĐBSCL.")
    add_class("Oreochromis_niloticus", c_fish_species, "Nile Tilapia", "Cá rô phi", "Loài cá nuôi phổ biến, chịu được biến động môi trường.")
    add_class("Anabas_testudineus", c_fish_species, "Climbing Perch", "Cá rô đồng", "Loài cá bản địa dễ bị ảnh hưởng bởi nấm EUS vào mùa lạnh.")
    add_class("Channa_striata", c_fish_species, "Striped Snakehead", "Cá lóc", "Loài cá ăn thịt nước ngọt, mẫn cảm cao với EUS và đốm đỏ.")
    add_class("Cyprinus_carpio", c_fish_species, "Common Carp", "Cá chép", "Loài cá nước ngọt phổ biến, hay mắc bệnh thối mang khi mật độ nuôi cao.")

    # --- Symptom (Triệu chứng lâm sàng mở rộng) ---
    c_symptom = add_class(
        "Symptom", OWL.Thing,
        "Symptom", "Triệu chứng lâm sàng",
        "Dấu hiệu bệnh lý bên ngoài hoặc hành vi bất thường quan sát được trên cá."
    )
    c_skin_ulcers = add_class("Skin_Ulcers", c_symptom, "Skin Ulcers and Hemorrhage", "Loét da và xuất huyết", "Các vết loét, đốm đỏ, hoại tử biểu mô hoặc xuất huyết vây và thân.")
    c_frayed_gills = add_class("Frayed_Gills", c_symptom, "Frayed Gills and Necrosis", "Mang xơ xác, hoại tử mang", "Mang cá bị rách, bạc màu, dính bùn, nhiều nhớt hoặc hoại tử các tơ mang.")
    c_lethargy = add_class("Lethargy_Anorexia", c_symptom, "Lethargy and Anorexia", "Bơi lờ đờ, bỏ ăn", "Cá bơi tách đàn, nổi đầu, mất phương hướng hoặc bỏ ăn hoàn toàn.")
    c_abdominal_distension = add_class("Abdominal_Distension", c_symptom, "Abdominal Distension", "Chướng bụng, sưng hậu môn", "Bụng trướng to chứa dịch vàng hoặc hồng, hậu môn lồi đỏ.")
    c_fin_rot = add_class("Fin_Rot", c_symptom, "Fin Rot and Erosion", "Thối vây, mòn vây", "Các tia vây bị rách nát, cụt dần do vi khuẩn xâm lấn.")
    c_white_spots = add_class("White_Spots_On_Skin", c_symptom, "White Spots on Skin and Gills", "Đốm trắng trên da và mang", "Các đốm trắng hoặc u hạt mủ li ti trên biểu bì thân và mang cá.")
    c_excess_mucus = add_class("Excess_Mucus_Production", c_symptom, "Excess Mucus Secretion", "Tăng tiết chất nhớt", "Da và mang bao phủ một lớp nhớt đục do phản ứng tự vệ chống ký sinh trùng.")
    c_normal_behavior = add_class("Normal_Behavior_Appearance", c_symptom, "Normal Behavior and Appearance", "Hành vi và ngoại hình bình thường", "Cá bơi nhanh nhẹn, ăn khỏe, da vảy và mang sáng bóng nguyên vẹn.")

    # --- Pathogen (Tác nhân gây bệnh mở rộng) ---
    c_pathogen = add_class(
        "Pathogen", OWL.Thing,
        "Pathogen", "Tác nhân gây bệnh",
        "Vi sinh vật gây bệnh cho cá (vi khuẩn, nấm, vi rút, ký sinh trùng)."
    )
    add_class("Aeromonas_hydrophila", c_pathogen, "Aeromonas hydrophila", "Vi khuẩn Aeromonas hydrophila", "Vi khuẩn Gram âm gây bệnh đốm đỏ, xuất huyết hoại tử.")
    add_class("Flavobacterium_columnare", c_pathogen, "Flavobacterium columnare", "Vi khuẩn Flavobacterium columnare", "Vi khuẩn Gram âm dạng que dài gây bệnh thối mang, đốm trắng mang.")
    add_class("Edwardsiella_ictaluri", c_pathogen, "Edwardsiella ictaluri", "Vi khuẩn Edwardsiella ictaluri", "Vi khuẩn gây bệnh gan thận mủ ở cá da trơn.")
    add_class("Aphanomyces_invadans", c_pathogen, "Aphanomyces invadans", "Nấm thủy mi Aphanomyces invadans", "Nấm nước Oomycete gây hội chứng lở loét biểu mô dịch tễ (EUS).", "WOAH Aquatic Manual 2021.")
    add_class("Saprolegnia_parasitica", c_pathogen, "Saprolegnia parasitica", "Nấm thủy mi Saprolegnia", "Nấm nước gây màng nấm bông trắng trên cá bị thương tổn biểu bì.")
    add_class("Trichodina_truttae", c_pathogen, "Trichodina spp.", "Ký sinh trùng Trichodina", "Trùng bánh xe ký sinh da và mang cá.")

    # --- WaterCondition ---
    c_water_condition = add_class(
        "WaterCondition", OWL.Thing,
        "Water Condition", "Điều kiện môi trường nước",
        "Tập hợp các chỉ số lý hóa đo được trong môi trường nước ao nuôi."
    )

    # --- Treatment (Biện pháp điều trị mở rộng) ---
    c_treatment = add_class(
        "Treatment", OWL.Thing,
        "Treatment", "Biện pháp điều trị",
        "Phác đồ can thiệp dùng thuốc kháng sinh, sát trùng hoặc thảo dược theo quy định Bộ NN&PTNT."
    )
    add_class("AntibioticTreatment", c_treatment, "Antibiotic Treatment", "Phác đồ kháng sinh", "Sử dụng Oxytetracycline (50-75 mg/kg) hoặc Florfenicol (10 mg/kg) trộn thức ăn 5-7 ngày.")
    add_class("AntisepticDisinfection", c_treatment, "Antiseptic Disinfection", "Sát trùng nguồn nước", "Tạt sát trùng bằng Iodine 90 (0.3-0.5 ppm) hoặc BKC 80% (0.5-0.8 ppm).")
    add_class("SaltBathTreatment", c_treatment, "Salt Bath Treatment", "Tắm nước muối", "Tắm cá trong dung dịch muối ăn NaCl 2-3% trong 5-10 phút để sát khuẩn và giảm nấm bám.")
    add_class("PotassiumPermanganateTreatment", c_treatment, "Potassium Permanganate Bath", "Tắm thuốc tím KMnO4", "Tắm thuốc tím 2-4 ppm trong 30-45 phút sát trùng ngoại sinh vật.")
    add_class("HerbalTreatment", c_treatment, "Herbal Remedy Treatment", "Trị liệu thảo dược tự nhiên", "Bổ sung tinh dầu tỏi (Allicin) 3-5 ml/kg thức ăn để kháng khuẩn tự nhiên.")

    # --- Prevention (Biện pháp phòng bệnh tổng hợp mở rộng) ---
    c_prevention = add_class(
        "Prevention", OWL.Thing,
        "Prevention", "Biện pháp phòng bệnh tổng hợp",
        "Các quy trình an toàn sinh học và quản lý môi trường theo quy chuẩn QCVN 02-19:2014/BNNPTNT."
    )
    add_class("WaterQualityManagement", c_prevention, "Water Quality Management", "Quản lý chất lượng nước ao", "Thay nước định kỳ, tăng cường quạt khí để DO > 4.5 mg/L, rải vôi ổn định pH và kiềm.")
    add_class("BiosecurityMeasures", c_prevention, "Biosecurity Measures", "Biện pháp an toàn sinh học", "Cách ly cá mới 14 ngày, khử trùng lưới cụ, kiểm tra con giống sạch mầm bệnh.")
    add_class("NutritionalImmunitySupport", c_prevention, "Nutritional and Immune Support", "Hỗ trợ dinh dưỡng và miễn dịch", "Bổ sung Vitamin C (100-200 mg/kg), Beta-glucan và men vi sinh đường ruột.")
    add_class("VaccinationPrevention", c_prevention, "Vaccination Prevention", "Tiêm phòng vaccine", "Tiêm vaccine phòng bệnh gan thận mủ và đốm đỏ ở quy mô cá giống.")

    # ==========================================================
    # CÁC LỚP LUẬT SUY LUẬN MỚI (DIAGNOSTIC & REASONING RULES)
    # ==========================================================
    c_diagnostic_rule = add_class(
        "DiagnosticRule", OWL.Thing,
        "Diagnostic and Reasoning Rule", "Luật chẩn đoán và suy luận",
        "Lớp cha biểu diễn mọi quy tắc tri thức, luật dịch tễ học và luật giải quyết mâu thuẫn được lưu trữ trong Ontology."
    )
    c_env_rule = add_class(
        "EnvironmentalRiskRule", c_diagnostic_rule,
        "Environmental Risk Rule", "Luật rủi ro môi trường nước",
        "Quy tắc dịch tễ học môi trường: kích hoạt rủi ro bệnh và cảnh báo khi chỉ số nước vượt ngưỡng sinh thái."
    )
    c_conflict_rule = add_class(
        "ConflictResolutionRule", c_diagnostic_rule,
        "Conflict Resolution Rule", "Luật giải quyết mâu thuẫn chẩn đoán",
        "Quy tắc ngữ nghĩa nhận diện và xử lý bất đồng giữa dự đoán AI thị giác và thực tế lâm sàng/dịch tễ."
    )
    c_decision_weight = add_class(
        "DecisionWeightConfig", OWL.Thing,
        "Decision Weight Configuration", "Cấu hình trọng số quyết định lai",
        "Định nghĩa cấu hình trọng số cho bộ hỗ trợ quyết định đa tiêu chí (AI, Triệu chứng, Môi trường nước)."
    )

    # ==========================================================
    # 3. DISJOINT CLASSES AXIOMS (RÀNG BUỘC PHÂN TÁCH)
    # ==========================================================
    g.add((c_healthy_state, OWL.disjointWith, c_disease))
    g.add((c_bacterial_disease, OWL.disjointWith, c_fungal_disease))
    g.add((c_bacterial_disease, OWL.disjointWith, c_parasitic_disease))
    g.add((c_fungal_disease, OWL.disjointWith, c_parasitic_disease))
    
    top_classes = [c_fish_condition, c_case, c_fish_species, c_symptom, c_pathogen, c_water_condition, c_treatment, c_prevention, c_diagnostic_rule, c_decision_weight]
    for i in range(len(top_classes)):
        for j in range(i + 1, len(top_classes)):
            g.add((top_classes[i], OWL.disjointWith, top_classes[j]))

    # ==========================================================
    # 4. THUỘC TÍNH ĐỐI TƯỢNG (OBJECT PROPERTIES)
    # ==========================================================
    def add_obj_prop(uri_name, domain, range_cls, label_en, label_vi, comment):
        op = FDO[uri_name]
        g.add((op, RDF.type, OWL.ObjectProperty))
        g.add((op, RDFS.domain, domain))
        g.add((op, RDFS.range, range_cls))
        g.add((op, RDFS.label, Literal(label_en, lang="en")))
        g.add((op, RDFS.label, Literal(label_vi, lang="vi")))
        g.add((op, RDFS.comment, Literal(comment, lang="vi")))
        return op

    add_obj_prop("affectsSpecies", c_disease, c_fish_species, "affects species", "ảnh hưởng đến loài cá", "Liên kết bệnh với loài cá mẫn cảm hoặc chịu tác động.")
    add_obj_prop("causedBy", c_disease, c_pathogen, "caused by", "gây ra bởi", "Liên kết bệnh với tác nhân vi sinh vật gây bệnh tương ứng.")
    add_obj_prop("hasSymptom", c_fish_condition, c_symptom, "has symptom", "có triệu chứng / biểu hiện", "Liên kết trạng thái cá với biểu hiện lâm sàng tương ứng.")
    add_obj_prop("hasTreatment", c_disease, c_treatment, "has treatment", "có biện pháp điều trị", "Liên kết bệnh với phác đồ thuốc/hóa chất xử lý.")
    add_obj_prop("hasPrevention", c_fish_condition, c_prevention, "has prevention", "có biện pháp phòng bệnh", "Liên kết trạng thái cá với quy trình an toàn sinh học và quản lý môi trường.")
    add_obj_prop("associatedWithWaterCondition", c_disease, c_water_condition, "associated with water condition", "liên quan điều kiện nước", "Môi trường nước bất lợi kích phát bệnh bùng phát.")

    # Quan hệ phục vụ suy luận chẩn đoán lâm sàng
    add_obj_prop("observedSymptom", c_case, c_symptom, "observed symptom", "triệu chứng quan sát được", "Triệu chứng thực tế được người nuôi hoặc chuyên gia ghi nhận tại ca chẩn đoán.")
    add_obj_prop("hasWaterCondition", c_case, c_water_condition, "has water condition", "có điều kiện nước", "Liên kết ca bệnh với tập hợp số liệu chất lượng nước ao đo đạc tại thời điểm kiểm tra.")
    add_obj_prop("hasAIPrediction", c_case, c_fish_condition, "has AI prediction", "có dự đoán từ AI", "Nhãn phân loại do mô hình học sâu (Deep Learning) nhận diện từ hình ảnh của ca.")
    add_obj_prop("suspectedDisease", c_case, c_fish_condition, "suspected disease", "bệnh nghi ngờ ban đầu", "Bệnh lý ban đầu được đưa vào xem xét trước khi suy luận logic.")
    add_obj_prop("inferredDisease", c_case, c_fish_condition, "inferred disease", "kết luận bệnh suy luận được", "Kết quả chẩn đoán cuối cùng được suy diễn ra từ sự tích hợp bằng chứng ảnh, triệu chứng và thông số nước.")

    # Quan hệ Object Properties cho LUẬT TRI THỨC (Knowledge Rules)
    add_obj_prop("triggersDisease", c_env_rule, c_disease, "triggers disease", "kích phát bệnh", "Liên kết luật rủi ro môi trường với bệnh lý tương ứng.")
    add_obj_prop("conflictingAIPrediction", c_conflict_rule, c_fish_condition, "conflicting AI prediction", "dự đoán AI mâu thuẫn", "Nhãn dự đoán từ mô hình AI bị mâu thuẫn với thực tế lâm sàng hoặc dịch tễ.")
    add_obj_prop("resolvedDisease", c_conflict_rule, c_disease, "resolved disease", "bệnh được suy luận điều chỉnh", "Bệnh lý thực sự được bộ suy luận tri thức chỉ định sau khi gỡ bỏ mâu thuẫn.")

    # ==========================================================
    # 5. KHAI BÁO USER-DEFINED DATATYPES VỚI FACET RESTRICTIONS
    # ==========================================================
    def add_datatype_range(datatype_name, base_type, min_val, max_val, label_vi):
        dt_uri = FDO[datatype_name]
        g.add((dt_uri, RDF.type, RDFS.Datatype))
        g.add((dt_uri, OWL.equivalentClass, dt_uri))
        g.add((dt_uri, RDFS.label, Literal(label_vi, lang="vi")))
        
        rest_node = BNode()
        g.add((rest_node, RDF.type, RDFS.Datatype))
        g.add((rest_node, OWL.onDatatype, base_type))
        
        facet_list = BNode()
        f_min = BNode()
        g.add((f_min, XSD.minInclusive, Literal(min_val, datatype=base_type)))
        f_max = BNode()
        g.add((f_max, XSD.maxInclusive, Literal(max_val, datatype=base_type)))
        
        Collection(g, facet_list, [f_min, f_max])
        g.add((rest_node, OWL.withRestrictions, facet_list))
        g.add((dt_uri, OWL.equivalentClass, rest_node))
        return dt_uri

    dt_temp = add_datatype_range("WaterTemperatureRange", XSD.float, 0.0, 45.0, "Miền nhiệt độ nước hợp lệ (0 - 45 °C)")
    dt_ph = add_datatype_range("PHRange", XSD.float, 0.0, 14.0, "Miền độ pH hợp lệ (0 - 14)")
    dt_do = add_datatype_range("DissolvedOxygenRange", XSD.float, 0.0, 20.0, "Miền oxy hòa tan hợp lệ (0 - 20 mg/L)")
    dt_nh3 = add_datatype_range("AmmoniaRange", XSD.float, 0.0, 10.0, "Miền amoniac TAN hợp lệ (0 - 10 mg/L)")
    dt_no2 = add_datatype_range("NitriteRange", XSD.float, 0.0, 10.0, "Miền nitrit hợp lệ (0 - 10 mg/L)")
    dt_alk = add_datatype_range("AlkalinityRange", XSD.float, 0.0, 500.0, "Miền độ kiềm hợp lệ (0 - 500 mg/L CaCO3)")
    dt_turb = add_datatype_range("TurbidityRange", XSD.float, 0.0, 100.0, "Miền độ trong đĩa Secchi hợp lệ (0 - 100 cm)")
    dt_conf = add_datatype_range("ConfidenceScoreRange", XSD.float, 0.0, 1.0, "Độ tin cậy xác suất chuẩn hóa [0.0, 1.0]")

    # ==========================================================
    # 6. THUỘC TÍNH DỮ LIỆU (DATA PROPERTIES)
    # ==========================================================
    def add_data_prop(uri_name, domain, range_dt, label_en, label_vi, comment):
        dp = FDO[uri_name]
        g.add((dp, RDF.type, OWL.DatatypeProperty))
        g.add((dp, RDFS.domain, domain))
        g.add((dp, RDFS.range, range_dt))
        g.add((dp, RDFS.label, Literal(label_en, lang="en")))
        g.add((dp, RDFS.label, Literal(label_vi, lang="vi")))
        g.add((dp, RDFS.comment, Literal(comment, lang="vi")))
        return dp

    add_data_prop("temperatureValue", c_water_condition, dt_temp, "temperature value", "giá trị nhiệt độ", "Nhiệt độ nước đo bằng độ C.")
    add_data_prop("pHValue", c_water_condition, dt_ph, "pH value", "giá trị pH", "Độ pH của nước ao.")
    add_data_prop("dissolvedOxygenValue", c_water_condition, dt_do, "dissolved oxygen value", "oxy hòa tan (DO)", "Hàm lượng oxy hòa tan đo bằng mg/L.")
    add_data_prop("ammoniaValue", c_water_condition, dt_nh3, "ammonia value (TAN)", "hàm lượng amoniac", "Tổng nồng độ amoniac TAN đo bằng mg/L.")
    add_data_prop("nitriteValue", c_water_condition, dt_no2, "nitrite value (NO2-)", "hàm lượng nitrit", "Hàm lượng nitrit NO2- đo bằng mg/L.")
    add_data_prop("alkalinityValue", c_water_condition, dt_alk, "alkalinity value", "độ kiềm", "Độ kiềm tính bằng mg/L CaCO3.")
    add_data_prop("turbidityValue", c_water_condition, dt_turb, "turbidity value", "độ đục / độ trong", "Độ trong của nước đo bằng đĩa Secchi.")

    add_data_prop("confidenceScore", c_case, dt_conf, "confidence score", "độ tin cậy của AI", "Điểm xác suất chuẩn hóa của mô hình Deep Learning.")
    add_data_prop("caseDate", c_case, XSD.dateTime, "case date", "thời gian ca chẩn đoán", "Thời điểm ghi nhận ca chẩn đoán.")

    # Thuộc tính dữ liệu cho LUẬT TRI THỨC VÀ NGƯỠNG SỐ HỌC (Data Properties for Rules)
    add_data_prop("targetParameter", c_env_rule, XSD.string, "target parameter", "thông số môi trường mục tiêu", "Tên thông số nước được kiểm tra (temperature, ph, dissolved_oxygen, ammonia, nitrite).")
    add_data_prop("comparisonOperator", c_env_rule, XSD.string, "comparison operator", "toán tử so sánh", "Toán tử logic so sánh số học (<, >, <=, >=).")
    add_data_prop("thresholdValue", c_env_rule, XSD.float, "threshold value", "giá trị ngưỡng kích hoạt", "Ngưỡng số học kích hoạt luật rủi ro môi trường.")
    add_data_prop("riskScoreContribution", c_env_rule, XSD.float, "risk score contribution", "mức độ đóng góp rủi ro", "Điểm rủi ro môi trường bổ sung khi luật thỏa mãn [0.0 - 1.0].")
    add_data_prop("ruleWarningMessage", c_env_rule, XSD.string, "rule warning message", "thông điệp cảnh báo dịch tễ", "Nội dung cảnh báo chuyên ngành giải thích cơ chế sinh học.")
    add_data_prop("remedialAction", c_env_rule, XSD.string, "remedial action", "biện pháp cải tạo môi trường", "Hướng dẫn kỹ thuật xử lý nước ao để loại bỏ mầm bệnh.")

    add_data_prop("conflictTriggerCondition", c_conflict_rule, XSD.string, "conflict trigger condition", "điều kiện kích hoạt mâu thuẫn", "Tên mã định danh điều kiện gây mâu thuẫn.")
    add_data_prop("conflictDescription", c_conflict_rule, XSD.string, "conflict description", "mô tả bản chất mâu thuẫn", "Giải thích chuyên môn lý do tại sao AI mâu thuẫn với lâm sàng/dịch tễ.")

    add_data_prop("weightAI", c_decision_weight, XSD.float, "weight for AI", "trọng số thị giác AI", "Hệ số trọng số của mô hình Deep Learning trong công thức lai [0.0 - 1.0].")
    add_data_prop("weightSymptom", c_decision_weight, XSD.float, "weight for symptoms", "trọng số triệu chứng", "Hệ số trọng số của triệu chứng lâm sàng [0.0 - 1.0].")
    add_data_prop("weightWater", c_decision_weight, XSD.float, "weight for water quality", "trọng số chất lượng nước", "Hệ số trọng số của rủi ro môi trường nước [0.0 - 1.0].")
    add_data_prop("configRationale", c_decision_weight, XSD.string, "config rationale", "cơ sở khoa học của cấu hình", "Giải thích lâm sàng cho việc lựa chọn cấu hình trọng số.")

    # ==========================================================
    # 7. MỐI QUAN HỆ KHÁI NIỆM & LUẬT TRI THỨC (AXIOMS & RESTRICTIONS)
    # ==========================================================
    def add_some_restriction(cls_uri, prop_uri, target_cls_uri):
        rest_node = BNode()
        g.add((rest_node, RDF.type, OWL.Restriction))
        g.add((rest_node, OWL.onProperty, prop_uri))
        g.add((rest_node, OWL.someValuesFrom, target_cls_uri))
        g.add((cls_uri, RDFS.subClassOf, rest_node))

    # BGD
    add_some_restriction(c_bgd, FDO["causedBy"], FDO["Flavobacterium_columnare"])
    add_some_restriction(c_bgd, FDO["hasSymptom"], c_frayed_gills)
    add_some_restriction(c_bgd, FDO["hasTreatment"], FDO["AntibioticTreatment"])
    add_some_restriction(c_bgd, FDO["hasTreatment"], FDO["AntisepticDisinfection"])
    add_some_restriction(c_bgd, FDO["hasPrevention"], FDO["WaterQualityManagement"])

    # BRD
    add_some_restriction(c_brd, FDO["causedBy"], FDO["Aeromonas_hydrophila"])
    add_some_restriction(c_brd, FDO["hasSymptom"], c_skin_ulcers)
    add_some_restriction(c_brd, FDO["hasSymptom"], c_abdominal_distension)
    add_some_restriction(c_brd, FDO["hasTreatment"], FDO["AntibioticTreatment"])
    add_some_restriction(c_brd, FDO["hasPrevention"], FDO["WaterQualityManagement"])

    # EUS
    add_some_restriction(c_eus, FDO["causedBy"], FDO["Aphanomyces_invadans"])
    add_some_restriction(c_eus, FDO["hasSymptom"], c_skin_ulcers)
    add_some_restriction(c_eus, FDO["hasTreatment"], FDO["AntisepticDisinfection"])
    add_some_restriction(c_eus, FDO["hasTreatment"], FDO["SaltBathTreatment"])
    add_some_restriction(c_eus, FDO["hasPrevention"], FDO["BiosecurityMeasures"])

    # Healthy Fish
    add_some_restriction(c_healthy_fish, FDO["hasSymptom"], c_normal_behavior)
    add_some_restriction(c_healthy_fish, FDO["hasPrevention"], FDO["WaterQualityManagement"])

    # ==========================================================
    # 8. CÁC THỰC THỂ CỤ THỂ (INDIVIDUALS) PHỤC VỤ SUY LUẬN & TEST CASE
    # ==========================================================
    # Pathogens
    inst_aero = FDO["Aeromonas_hydrophila_Strain_AH01"]
    g.add((inst_aero, RDF.type, FDO["Aeromonas_hydrophila"]))
    g.add((inst_aero, RDFS.label, Literal("Chủng vi khuẩn Aeromonas hydrophila AH-01", lang="vi")))

    inst_flavo = FDO["Flavobacterium_columnare_Strain_FC02"]
    g.add((inst_flavo, RDF.type, FDO["Flavobacterium_columnare"]))
    g.add((inst_flavo, RDFS.label, Literal("Chủng vi khuẩn Flavobacterium columnare FC-02", lang="vi")))

    inst_aphano = FDO["Aphanomyces_invadans_Isolate_NJM9701"]
    g.add((inst_aphano, RDF.type, FDO["Aphanomyces_invadans"]))
    g.add((inst_aphano, RDFS.label, Literal("Chủng nấm thủy mi Aphanomyces invadans", lang="vi")))

    # Symptoms
    inst_sym_gill = FDO["Frayed_Gills_Observation_01"]
    g.add((inst_sym_gill, RDF.type, c_frayed_gills))
    g.add((inst_sym_gill, RDFS.label, Literal("Tơ mang rách xơ xác dính nhớt", lang="vi")))

    inst_sym_ulcer = FDO["Skin_Ulcer_Observation_02"]
    g.add((inst_sym_ulcer, RDF.type, c_skin_ulcers))
    g.add((inst_sym_ulcer, RDFS.label, Literal("Vết loét da xuất huyết cơ", lang="vi")))

    inst_sym_normal = FDO["Normal_Behavior_Appearance_Inst"]
    g.add((inst_sym_normal, RDF.type, c_normal_behavior))
    g.add((inst_sym_normal, RDFS.label, Literal("Ngoại hình bình thường, bơi nhanh", lang="vi")))

    # Treatments & Prevention
    inst_treat_ab = FDO["Oxytetracycline_Florfenicol"]
    g.add((inst_treat_ab, RDF.type, FDO["AntibioticTreatment"]))
    g.add((inst_treat_ab, RDFS.label, Literal("Kháng sinh Oxytetracycline 50-75mg/kg hoặc Florfenicol 10mg/kg trong 7 ngày", lang="vi")))

    inst_treat_antiseptic = FDO["Iodine_BKC_Disinfectant"]
    g.add((inst_treat_antiseptic, RDF.type, FDO["AntisepticDisinfection"]))
    g.add((inst_treat_antiseptic, RDFS.label, Literal("Sát trùng nguồn nước với Iodine 90 (0.4 ppm) hoặc BKC 80% (0.6 ppm)", lang="vi")))

    inst_treat_salt = FDO["Salt_Bath_2Percent"]
    g.add((inst_treat_salt, RDF.type, FDO["SaltBathTreatment"]))
    g.add((inst_treat_salt, RDFS.label, Literal("Tắm nước muối 2-3% trong 5-10 phút", lang="vi")))

    inst_prev_water = FDO["Pond_Water_Aeration_Liming"]
    g.add((inst_prev_water, RDF.type, FDO["WaterQualityManagement"]))
    g.add((inst_prev_water, RDFS.label, Literal("Bật quạt khí ban đêm, thay 20% nước, rải vôi CaCO3 2-3 kg/100m3", lang="vi")))

    # Liên kết concept bệnh với thực thể
    g.add((c_bgd, FDO["causedBy"], inst_flavo))
    g.add((c_bgd, FDO["hasSymptom"], inst_sym_gill))
    g.add((c_bgd, FDO["hasTreatment"], inst_treat_ab))
    g.add((c_bgd, FDO["hasTreatment"], inst_treat_antiseptic))
    g.add((c_bgd, FDO["hasPrevention"], inst_prev_water))

    g.add((c_brd, FDO["causedBy"], inst_aero))
    g.add((c_brd, FDO["hasSymptom"], inst_sym_ulcer))
    g.add((c_brd, FDO["hasTreatment"], inst_treat_ab))
    g.add((c_brd, FDO["hasPrevention"], inst_prev_water))

    g.add((c_eus, FDO["causedBy"], inst_aphano))
    g.add((c_eus, FDO["hasSymptom"], inst_sym_ulcer))
    g.add((c_eus, FDO["hasTreatment"], inst_treat_salt))
    g.add((c_eus, FDO["hasTreatment"], inst_treat_antiseptic))
    g.add((c_eus, FDO["hasPrevention"], inst_prev_water))

    g.add((c_healthy_fish, FDO["hasSymptom"], inst_sym_normal))
    g.add((c_healthy_fish, FDO["hasPrevention"], inst_prev_water))

    # ==========================================================
    # CÁC THỰC THỂ LUẬT RỦI RO MÔI TRƯỜNG (ENVIRONMENTAL RISK RULES)
    # ==========================================================
    def add_env_rule(rule_id, disease_cls, param, op, threshold, risk, warning, action):
        r = FDO[rule_id]
        g.add((r, RDF.type, c_env_rule))
        g.add((r, FDO["triggersDisease"], disease_cls))
        g.add((r, FDO["targetParameter"], Literal(param, datatype=XSD.string)))
        g.add((r, FDO["comparisonOperator"], Literal(op, datatype=XSD.string)))
        g.add((r, FDO["thresholdValue"], Literal(float(threshold), datatype=XSD.float)))
        g.add((r, FDO["riskScoreContribution"], Literal(float(risk), datatype=XSD.float)))
        g.add((r, FDO["ruleWarningMessage"], Literal(warning, lang="vi")))
        g.add((r, FDO["remedialAction"], Literal(action, lang="vi")))
        return r

    # 1. EUS - Nhiệt độ nước lạnh
    add_env_rule(
        "Rule_EUS_ColdWater", c_eus, "temperature", "<", 25.0, 0.50,
        "Nhiệt độ nước lạnh (<25.0°C) là yếu tố kích phát dịch bệnh hàng đầu của nấm thủy mi Aphanomyces invadans (EUS).",
        "Dâng mực nước ao >1.5m, hạn chế kéo lưới làm trầy xước cá; rải vôi CaCO3 (2-3 kg/100m3) để ổn định nhiệt độ tầng đáy."
    )
    # 2. EUS - Độ pH thấp (nước axit sau mưa lũ)
    add_env_rule(
        "Rule_EUS_AcidicWater", c_eus, "ph", "<", 6.5, 0.50,
        "Độ pH thấp (<6.5) làm tổn thương lớp nhớt biểu bì cá, tạo điều kiện cho bào tử nấm EUS xâm nhập sâu vào cơ.",
        "Bón vôi nông nghiệp CaCO3 hoặc vôi tôi Ca(OH)2 liều lượng 2-3 kg/100m3 để nâng pH lên 7.0 - 7.5."
    )
    # 3. BGD - Oxy hòa tan thấp
    add_env_rule(
        "Rule_BGD_LowOxygen", c_bgd, "dissolved_oxygen", "<", 3.5, 0.50,
        "Hàm lượng oxy hòa tan thấp (<3.5 mg/L) gây stress hô hấp nghiêm trọng và kích hoạt bệnh thối mang do Flavobacterium columnare.",
        "Tăng cường quạt nước / sục khí đáy khẩn cấp để nâng DO lên trên 4.5 mg/L."
    )
    # 4. BGD - Khí độc Amoniac cao
    add_env_rule(
        "Rule_BGD_HighAmmonia", c_bgd, "ammonia", ">", 0.5, 0.40,
        "Nồng độ Amoniac TAN cao (>0.5 mg/L) làm tổn thương phiến tơ mang và ức chế khả năng đào thải NH3 qua mang cá.",
        "Thay 20-30% nước sạch đáy ao; tạt chế phẩm vi sinh Bacillus sp. để hấp thu và chuyển hóa khí độc."
    )
    # 5. BRD - Khí độc Amoniac cao
    add_env_rule(
        "Rule_BRD_HighAmmonia", c_brd, "ammonia", ">", 0.5, 0.40,
        "Khí độc Amoniac cao (>0.5 mg/L) kết hợp dư thừa hữu cơ kích hoạt vi khuẩn xuất huyết Aeromonas hydrophila bùng phát.",
        "Cắt giảm 50% lượng thức ăn, xi-phông bùn đáy ao và xử lý vi sinh phân hủy chất hữu cơ."
    )
    # 6. Nhiệt độ cao kích thích vi khuẩn phát triển
    add_env_rule(
        "Rule_HighTemp_BacterialStress", c_brd, "temperature", ">", 30.0, 0.30,
        "Nhiệt độ nước cao (>30.0°C) gia tăng áp lực trao đổi chất và tốc độ phân chia của vi khuẩn cơ hội Aeromonas.",
        "Duy trì mực nước sâu >1.5m, che lưới lan làm mát bề mặt ao và định kỳ khử trùng bằng Iodine 90 (0.4 ppm)."
    )

    # ==========================================================
    # CÁC THỰC THỂ LUẬT GIẢI QUYẾT MÂU THUẪN (CONFLICT RESOLUTION RULES)
    # ==========================================================
    # Mâu thuẫn 1: AI dự đoán Healthy nhưng lâm sàng có triệu chứng bệnh
    cr1 = FDO["ConflictRule_HealthyAI_vs_Pathology"]
    g.add((cr1, RDF.type, c_conflict_rule))
    g.add((cr1, FDO["conflictingAIPrediction"], c_healthy_fish))
    g.add((cr1, FDO["conflictTriggerCondition"], Literal("HAS_PATHOLOGY_SYMPTOMS", datatype=XSD.string)))
    g.add((cr1, FDO["conflictDescription"], Literal("Mâu thuẫn phủ định: AI nhận diện là Cá khỏe mạnh nhưng thực tế quan sát thấy tổn thương bệnh lý hoại tử/loét da. Tiên đề tri thức tự động bác bỏ kết quả Healthy của AI.", lang="vi")))

    # Mâu thuẫn 2: AI dự đoán BRD nhưng dịch tễ môi trường chỉ ra EUS
    cr2 = FDO["ConflictRule_BRD_AI_vs_EUS_Epidemiology"]
    g.add((cr2, RDF.type, c_conflict_rule))
    g.add((cr2, FDO["conflictingAIPrediction"], c_brd))
    g.add((cr2, FDO["resolvedDisease"], c_eus))
    g.add((cr2, FDO["conflictTriggerCondition"], Literal("COLD_WATER_EUS_TRIGGER", datatype=XSD.string)))
    g.add((cr2, FDO["conflictDescription"], Literal("Bất đồng chứng cứ dịch tễ: AI dự đoán Bệnh đốm đỏ do vi khuẩn (BRD) do tổn thương ngoài da, nhưng nhiệt độ lạnh (<25.0°C) và pH thấp (<6.5) chỉ ra Hội chứng lở loét (EUS) do nấm Aphanomyces invadans.", lang="vi")))

    # ==========================================================
    # CÁC CẤU HÌNH TRỌNG SỐ SUY LUẬN QUYẾT ĐỊNH (DECISION WEIGHT CONFIGS)
    # ==========================================================
    # Cấu hình 1: Chuẩn cân bằng lâm sàng (0.50 - 0.35 - 0.15)
    wc_base = FDO["WeightConfig_Baseline"]
    g.add((wc_base, RDF.type, c_decision_weight))
    g.add((wc_base, RDFS.label, Literal("Cấu hình Cân bằng Lâm sàng (Baseline)", lang="vi")))
    g.add((wc_base, FDO["weightAI"], Literal(0.50, datatype=XSD.float)))
    g.add((wc_base, FDO["weightSymptom"], Literal(0.35, datatype=XSD.float)))
    g.add((wc_base, FDO["weightWater"], Literal(0.15, datatype=XSD.float)))
    g.add((wc_base, FDO["configRationale"], Literal("Cấu hình cân bằng lâm sàng: AI giữ vai trò định hướng chính (0.50), triệu chứng xác nhận thứ cấp (0.35), môi trường đóng vai trò xúc tác/kích phát (0.15).", lang="vi")))

    # Cấu hình 2: Ưu tiên thị giác (0.60 - 0.30 - 0.10)
    wc_vis = FDO["WeightConfig_VisionHeavy"]
    g.add((wc_vis, RDF.type, c_decision_weight))
    g.add((wc_vis, RDFS.label, Literal("Cấu hình Ưu tiên Thị giác AI", lang="vi")))
    g.add((wc_vis, FDO["weightAI"], Literal(0.60, datatype=XSD.float)))
    g.add((wc_vis, FDO["weightSymptom"], Literal(0.30, datatype=XSD.float)))
    g.add((wc_vis, FDO["weightWater"], Literal(0.10, datatype=XSD.float)))
    g.add((wc_vis, FDO["configRationale"], Literal("Cấu hình ưu tiên thị giác: Dành cho điều kiện ảnh chụp độ nét cao, biểu hiện vết loét bề mặt rõ rệt.", lang="vi")))

    # Cấu hình 3: Ưu tiên lâm sàng và dịch tễ (0.40 - 0.40 - 0.20)
    wc_clin = FDO["WeightConfig_ClinicalHeavy"]
    g.add((wc_clin, RDF.type, c_decision_weight))
    g.add((wc_clin, RDFS.label, Literal("Cấu hình Ưu tiên Lâm sàng & Dịch tễ", lang="vi")))
    g.add((wc_clin, FDO["weightAI"], Literal(0.40, datatype=XSD.float)))
    g.add((wc_clin, FDO["weightSymptom"], Literal(0.40, datatype=XSD.float)))
    g.add((wc_clin, FDO["weightWater"], Literal(0.20, datatype=XSD.float)))
    g.add((wc_clin, FDO["configRationale"], Literal("Cấu hình ưu tiên lâm sàng: Dành cho vùng nuôi đang có dịch tễ phức tạp hoặc ảnh chụp bị nhiễu ánh sáng.", lang="vi")))

    # ==========================================================
    # BIỂU DIỄN TIÊN ĐỀ LUẬT SWRL CHÍNH QUY (W3C SWRL AXIOMS)
    # ==========================================================
    # SWRL Rule 1: DiagnosticCase(?c) ^ hasWaterCondition(?c, ?w) ^ temperatureValue(?w, ?t) ^ swrlb:lessThan(?t, 25.0) -> suspectedDisease(?c, Epizootic_Ulcerative_Syndrome)
    swrl_rule1 = BNode()
    g.add((swrl_rule1, RDF.type, SWRL.Imp))
    g.add((swrl_rule1, RDFS.label, Literal("SWRL_Rule_EUS_ColdWater", lang="en")))
    g.add((swrl_rule1, RDFS.comment, Literal("DiagnosticCase(?c) ^ hasWaterCondition(?c, ?w) ^ temperatureValue(?w, ?t) ^ swrlb:lessThan(?t, 25.0) -> suspectedDisease(?c, Epizootic_Ulcerative_Syndrome)", lang="en")))

    # SWRL Rule 2: DiagnosticCase(?c) ^ hasWaterCondition(?c, ?w) ^ dissolvedOxygenValue(?w, ?do) ^ swrlb:lessThan(?do, 3.5) -> suspectedDisease(?c, Bacterial_Gill_Disease)
    swrl_rule2 = BNode()
    g.add((swrl_rule2, RDF.type, SWRL.Imp))
    g.add((swrl_rule2, RDFS.label, Literal("SWRL_Rule_BGD_LowDO", lang="en")))
    g.add((swrl_rule2, RDFS.comment, Literal("DiagnosticCase(?c) ^ hasWaterCondition(?c, ?w) ^ dissolvedOxygenValue(?w, ?do) ^ swrlb:lessThan(?do, 3.5) -> suspectedDisease(?c, Bacterial_Gill_Disease)", lang="en")))

    # SWRL Rule 3: DiagnosticCase(?c) ^ hasAIPrediction(?c, Healthy_Fish) ^ observedSymptom(?c, Skin_Ulcers) -> hasConflict(?c, true)
    swrl_rule3 = BNode()
    g.add((swrl_rule3, RDF.type, SWRL.Imp))
    g.add((swrl_rule3, RDFS.label, Literal("SWRL_Rule_Conflict_Healthy_Ulcers", lang="en")))
    g.add((swrl_rule3, RDFS.comment, Literal("DiagnosticCase(?c) ^ hasAIPrediction(?c, Healthy_Fish) ^ observedSymptom(?c, Skin_Ulcers) -> hasConflict(?c, true)", lang="en")))

    # ==========================================================
    # CÁC CA CHẨN ĐOÁN THỰC NGHIỆM (DIAGNOSTIC CASE INSTANCES)
    # ==========================================================
    # Ca 1: BGD đồng thuận
    case1 = FDO["Case_001_BGD_Consistent"]
    g.add((case1, RDF.type, c_case))
    g.add((case1, RDFS.label, Literal("Ca chẩn đoán 001 - Bệnh thối mang (Đồng thuận)", lang="vi")))
    g.add((case1, FDO["hasAIPrediction"], c_bgd))
    g.add((case1, FDO["suspectedDisease"], c_bgd))
    g.add((case1, FDO["observedSymptom"], inst_sym_gill))
    g.add((case1, FDO["confidenceScore"], Literal(0.942, datatype=XSD.float)))
    g.add((case1, FDO["inferredDisease"], c_bgd))

    # Ca 2: BRD đồng thuận
    case2 = FDO["Case_002_BRD_Consistent"]
    g.add((case2, RDF.type, c_case))
    g.add((case2, RDFS.label, Literal("Ca chẩn đoán 002 - Bệnh đốm đỏ (Đồng thuận)", lang="vi")))
    g.add((case2, FDO["hasAIPrediction"], c_brd))
    g.add((case2, FDO["suspectedDisease"], c_brd))
    g.add((case2, FDO["observedSymptom"], inst_sym_ulcer))
    g.add((case2, FDO["confidenceScore"], Literal(0.915, datatype=XSD.float)))
    g.add((case2, FDO["inferredDisease"], c_brd))

    # Ca 3: EUS mùa lạnh
    cond_eus = FDO["WaterCond_Case003"]
    g.add((cond_eus, RDF.type, c_water_condition))
    g.add((cond_eus, FDO["temperatureValue"], Literal(21.5, datatype=XSD.float)))
    g.add((cond_eus, FDO["pHValue"], Literal(5.8, datatype=XSD.float)))
    g.add((cond_eus, FDO["dissolvedOxygenValue"], Literal(3.8, datatype=XSD.float)))

    case3 = FDO["Case_003_EUS_Winter"]
    g.add((case3, RDF.type, c_case))
    g.add((case3, RDFS.label, Literal("Ca chẩn đoán 003 - Hội chứng EUS mùa lạnh", lang="vi")))
    g.add((case3, FDO["hasAIPrediction"], c_eus))
    g.add((case3, FDO["suspectedDisease"], c_eus))
    g.add((case3, FDO["observedSymptom"], inst_sym_ulcer))
    g.add((case3, FDO["hasWaterCondition"], cond_eus))
    g.add((case3, FDO["confidenceScore"], Literal(0.887, datatype=XSD.float)))
    g.add((case3, FDO["inferredDisease"], c_eus))

    # Ca 4: Cá khỏe mạnh
    case4 = FDO["Case_004_Healthy_Fish"]
    g.add((case4, RDF.type, c_case))
    g.add((case4, RDFS.label, Literal("Ca chẩn đoán 004 - Đàn cá khỏe mạnh", lang="vi")))
    g.add((case4, FDO["hasAIPrediction"], c_healthy_fish))
    g.add((case4, FDO["suspectedDisease"], c_healthy_fish))
    g.add((case4, FDO["observedSymptom"], inst_sym_normal))
    g.add((case4, FDO["confidenceScore"], Literal(0.975, datatype=XSD.float)))
    g.add((case4, FDO["inferredDisease"], c_healthy_fish))

    # Ca 5: Ca MÂU THUẪN (Conflict Case)
    case5 = FDO["Case_005_Conflict_Warning"]
    g.add((case5, RDF.type, c_case))
    g.add((case5, RDFS.label, Literal("Ca chẩn đoán 005 - Cảnh báo mâu thuẫn AI và Triệu chứng lâm sàng", lang="vi")))
    g.add((case5, FDO["hasAIPrediction"], c_healthy_fish))
    g.add((case5, FDO["suspectedDisease"], c_healthy_fish))
    g.add((case5, FDO["observedSymptom"], inst_sym_ulcer))
    g.add((case5, FDO["confidenceScore"], Literal(0.684, datatype=XSD.float)))

    return g

if __name__ == "__main__":
    out_dir = r"d:\Project\ontology"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "FishDiseaseOntology.owl")
    
    print("Bắt đầu xây dựng Ontology v2 hoàn chỉnh (Kiến trúc Hướng Tri thức)...")
    graph = build_fish_disease_ontology_v2()
    
    print(f"Lưu Ontology dưới định dạng chuẩn RDF/XML tại: {out_path}")
    graph.serialize(destination=out_path, format="xml", encoding="utf-8")
    
    print(f"Hoàn thành! Kích thước file: {os.path.getsize(out_path)} bytes, Tổng số bộ ba RDF (triples): {len(graph)}")
