# BÁO CÁO KẾT QUẢ KIỂM THỬ ONTOLOGY V2 VÀ 5 CÂU HỎI NĂNG LỰC (CQ1 - CQ5)

**Đề tài:** Hệ thống hỗ trợ chẩn đoán bệnh cá kết hợp tri thức Ontology và học máy
**Sinh viên thực hiện:** Trương Thị Kim Yên - Ngành Hệ thống Thông tin Quản lý, ĐH Nha Trang
**File Ontology:** `d:\Project\ontology\FishDiseaseOntology.owl` (Định dạng chuẩn W3C RDF/XML)
**Ontology IRI:** `http://www.semanticweb.org/fish-disease-ontology#`
**HermiT Reasoner Status:** **CONSISTENT (NHẤT QUÁN TUYỆT ĐỐI - KHÔNG MÂU THUẪN)**

---

### CQ1: Những bệnh nào do vi khuẩn gây ra và có triệu chứng xuất huyết hoặc loét da (Skin Ulcers)?

```sparql
PREFIX fdo: <http://www.semanticweb.org/fish-disease-ontology#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                SELECT DISTINCT ?disease ?diseaseLabel ?symptomLabel WHERE {
                    ?disease rdfs:subClassOf* fdo:BacterialDisease .
                    ?disease rdfs:label ?diseaseLabel .
                    FILTER(LANG(?diseaseLabel) = 'vi')
                    {
                        ?disease fdo:hasSymptom ?sym .
                        ?sym rdfs:label ?symptomLabel .
                        FILTER(LANG(?symptomLabel) = 'vi')
                    } UNION {
                        ?disease rdfs:subClassOf ?rest .
                        ?rest owl:onProperty fdo:hasSymptom .
                        ?rest owl:someValuesFrom ?symClass .
                        ?symClass rdfs:label ?symptomLabel .
                        FILTER(LANG(?symptomLabel) = 'vi')
                    }
                }
```

**Kết quả truy vấn SPARQL:**

- disease: http://www.semanticweb.org/fish-disease-ontology#Bacterial_Gill_Disease | diseaseLabel: Bệnh thối mang do vi khuẩn | symptomLabel: Tơ mang rách xơ xác dính nhớt  
- disease: http://www.semanticweb.org/fish-disease-ontology#Bacterial_Gill_Disease | diseaseLabel: Bệnh thối mang do vi khuẩn | symptomLabel: Mang xơ xác, hoại tử mang  
- disease: http://www.semanticweb.org/fish-disease-ontology#Bacterial_Red_Disease | diseaseLabel: Bệnh đốm đỏ do vi khuẩn | symptomLabel: Vết loét da xuất huyết cơ  
- disease: http://www.semanticweb.org/fish-disease-ontology#Bacterial_Red_Disease | diseaseLabel: Bệnh đốm đỏ do vi khuẩn | symptomLabel: Chướng bụng, sưng hậu môn  
- disease: http://www.semanticweb.org/fish-disease-ontology#Bacterial_Red_Disease | diseaseLabel: Bệnh đốm đỏ do vi khuẩn | symptomLabel: Loét da và xuất huyết  


### CQ2: Hội chứng lở loét dịch tễ (EUS) do tác nhân sinh học nào gây ra và thuộc phân loại bệnh nào?

```sparql
PREFIX fdo: <http://www.semanticweb.org/fish-disease-ontology#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX dc: <http://purl.org/dc/elements/1.1/>
                SELECT DISTINCT ?diseaseLabel ?pathogenLabel ?parentClassLabel ?source WHERE {
                    fdo:Epizootic_Ulcerative_Syndrome rdfs:label ?diseaseLabel .
                    FILTER(LANG(?diseaseLabel) = 'vi')
                    fdo:Epizootic_Ulcerative_Syndrome rdfs:subClassOf ?parent .
                    ?parent rdfs:label ?parentClassLabel .
                    FILTER(LANG(?parentClassLabel) = 'vi')
                    OPTIONAL {
                        fdo:Epizootic_Ulcerative_Syndrome fdo:causedBy ?pathogen .
                        ?pathogen rdfs:label ?pathogenLabel .
                        FILTER(LANG(?pathogenLabel) = 'vi')
                    }
                    OPTIONAL { fdo:Epizootic_Ulcerative_Syndrome dc:source ?source }
                }
```

**Kết quả truy vấn SPARQL:**

- diseaseLabel: Hội chứng lở loét dịch tễ (EUS) | pathogenLabel: Chủng nấm thủy mi Aphanomyces invadans | parentClassLabel: Bệnh do nấm và nấm nước | source: FAO Fisheries Technical Paper No. 402, Lilley et al. (1997); WOAH Manual of Diagnostic Tests for Aquatic Animals (2021), Chapter 2.3.2.  


### CQ3: Phác đồ điều trị và biện pháp phòng ngừa cho bệnh thối mang (Bacterial Gill Disease) gồm những gì?

```sparql
PREFIX fdo: <http://www.semanticweb.org/fish-disease-ontology#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                SELECT DISTINCT ?treatmentLabel ?preventionLabel WHERE {
                    fdo:Bacterial_Gill_Disease fdo:hasTreatment ?treat .
                    ?treat rdfs:label ?treatmentLabel .
                    FILTER(LANG(?treatmentLabel) = 'vi')
                    OPTIONAL {
                        fdo:Bacterial_Gill_Disease fdo:hasPrevention ?prev .
                        ?prev rdfs:label ?preventionLabel .
                        FILTER(LANG(?preventionLabel) = 'vi')
                    }
                }
```

**Kết quả truy vấn SPARQL:**

- treatmentLabel: Kháng sinh Oxytetracycline 50-75mg/kg hoặc Florfenicol 10mg/kg trong 7 ngày | preventionLabel: Bật quạt khí ban đêm, thay 20% nước, rải vôi CaCO3 2-3 kg/100m3  
- treatmentLabel: Sát trùng nguồn nước với Iodine 90 (0.4 ppm) hoặc BKC 80% (0.6 ppm) | preventionLabel: Bật quạt khí ban đêm, thay 20% nước, rải vôi CaCO3 2-3 kg/100m3  


### CQ4: Với ca chẩn đoán lâm sàng Case_001 có triệu chứng tơ mang rách xơ xác, kết quả AI dự đoán là gì và mức độ tin cậy bao nhiêu?

```sparql
PREFIX fdo: <http://www.semanticweb.org/fish-disease-ontology#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                SELECT ?case ?aiPredLabel ?confidence ?symptomLabel WHERE {
                    VALUES ?case { fdo:Case_001_BGD_Consistent }
                    ?case fdo:hasAIPrediction ?aiPred .
                    ?aiPred rdfs:label ?aiPredLabel .
                    FILTER(LANG(?aiPredLabel) = 'vi')
                    ?case fdo:confidenceScore ?confidence .
                    ?case fdo:observedSymptom ?sym .
                    ?sym rdfs:label ?symptomLabel .
                    FILTER(LANG(?symptomLabel) = 'vi')
                }
```

**Kết quả truy vấn SPARQL:**

- case: http://www.semanticweb.org/fish-disease-ontology#Case_001_BGD_Consistent | aiPredLabel: Bệnh thối mang do vi khuẩn | confidence: 0.942 | symptomLabel: Tơ mang rách xơ xác dính nhớt  


### CQ5: Phát hiện mâu thuẫn (Conflict Detection): Những ca chẩn đoán nào AI nhận diện là 'Cá khỏe mạnh' nhưng thực tế lại quan sát thấy triệu chứng tổn thương bệnh lý?

```sparql
PREFIX fdo: <http://www.semanticweb.org/fish-disease-ontology#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                SELECT ?case ?caseLabel ?aiPredLabel ?symptomLabel ?confidence WHERE {
                    ?case a fdo:DiagnosticCase .
                    ?case rdfs:label ?caseLabel .
                    ?case fdo:hasAIPrediction fdo:Healthy_Fish .
                    fdo:Healthy_Fish rdfs:label ?aiPredLabel .
                    FILTER(LANG(?aiPredLabel) = 'vi')
                    ?case fdo:observedSymptom ?sym .
                    ?sym a ?symClass .
                    FILTER(?symClass != fdo:Normal_Behavior_Appearance)
                    ?sym rdfs:label ?symptomLabel .
                    FILTER(LANG(?symptomLabel) = 'vi')
                    ?case fdo:confidenceScore ?confidence .
                }
```

**Kết quả truy vấn SPARQL:**

- case: http://www.semanticweb.org/fish-disease-ontology#Case_005_Conflict_Warning | caseLabel: Ca chẩn đoán 005 - Cảnh báo mâu thuẫn AI và Triệu chứng lâm sàng | aiPredLabel: Cá khỏe mạnh | symptomLabel: Vết loét da xuất huyết cơ | confidence: 0.684  

