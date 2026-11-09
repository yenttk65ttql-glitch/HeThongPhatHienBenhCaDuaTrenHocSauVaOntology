# -*- coding: utf-8 -*-
"""
Module Dự Đoán Thị Giác Máy Tính (Deep Learning Predictor)
Đề tài: Hệ thống Hỗ trợ Chẩn đoán Bệnh Cá Dựa trên Học Sâu và Ontology
Sinh viên: Trương Thị Kim Yên - ĐH Nha Trang

Chức năng:
- Tải mô hình học sâu đã chốt (EfficientNet-B0) từ checkpoint đã đóng băng.
- Thực hiện tiền xử lý ảnh đầu vào chuẩn hóa (Resize 224x224, Normalize ImageNet).
- Trích xuất phân phối xác suất Softmax và Top-k (mặc định Top-3) class probabilities.
- Trả về tensor phục vụ trực tiếp cho mô-đun giải thích Grad-CAM.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Union, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np

class FishDiseasePredictor:
    """
    Bộ suy diễn phân loại hình ảnh bệnh cá sử dụng mạng EfficientNet-B0 đã chốt.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None, 
                 class_to_idx_path: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Khởi tạo mô hình và tải trọng số đã huấn luyện tối ưu.
        """
        project_root = Path(__file__).resolve().parent.parent
        
        # 1. Xác định thiết bị tính toán
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # 2. Xác định đường dẫn class_to_idx.json
        if class_to_idx_path is None or not os.path.exists(class_to_idx_path):
            candidates = [
                class_to_idx_path,
                project_root / "models" / "class_to_idx.json",
                project_root / "class_to_idx.json",
                "models/class_to_idx.json"
            ]
            for c in candidates:
                if c and os.path.exists(c):
                    class_to_idx_path = str(c)
                    break
                    
        with open(class_to_idx_path, "r", encoding="utf-8") as f:
            self.class_to_idx = json.load(f)
            
        # Đảo ngược mapping: idx -> class_name
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)
        
        # 3. Xác định đường dẫn model checkpoint
        if model_path is None or not os.path.exists(model_path):
            candidates = [
                model_path,
                project_root / "models" / "efficientnet_b0_best.pth",
                project_root / "efficientnet_b0_best.pth",
                "models/efficientnet_b0_best.pth"
            ]
            for c in candidates:
                if c and os.path.exists(c):
                    model_path = str(c)
                    break
                    
        self.model_path = str(model_path)
        
        # 4. Khởi tạo kiến trúc EfficientNet-B0
        self.model = models.efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, self.num_classes)
        
        # Nạp state_dict
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # 5. Pipeline tiền xử lý ảnh chuẩn
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def preprocess_image(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> Tuple[torch.Tensor, Image.Image]:
        """
        Tiền xử lý ảnh đầu vào thành PyTorch Tensor [1, 3, 224, 224] và PIL Image gốc.
        """
        if isinstance(image_input, (str, Path)):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Không tìm thấy tệp ảnh tại: {image_input}")
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input).convert("RGB")
        else:
            raise TypeError(f"Định dạng ảnh không được hỗ trợ: {type(image_input)}")
            
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        return tensor, pil_img
        
    def predict(self, 
                image_input: Union[str, Path, Image.Image, np.ndarray], 
                top_k: int = 3) -> Dict[str, Any]:
        """
        Thực hiện dự đoán phân loại bệnh từ ảnh, trả về Top-k nhãn và xác suất.
        """
        start_time = time.time()
        tensor, pil_img = self.preprocess_image(image_input)
        
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
            
        inference_time_ms = round((time.time() - start_time) * 1000.0, 2)
        
        # Xếp hạng tất cả các lớp theo xác suất giảm dần
        sorted_indices = np.argsort(probs)[::-1]
        
        top_predictions = []
        for rank, idx in enumerate(sorted_indices[:top_k], 1):
            class_name = self.idx_to_class[idx]
            conf = float(probs[idx])
            top_predictions.append({
                "rank": rank,
                "class_name": class_name,
                "confidence": round(conf, 4),
                "confidence_percent": f"{conf * 100.0:.2f}%",
                "class_idx": int(idx)
            })
            
        all_probabilities = {
            self.idx_to_class[idx]: round(float(probs[idx]), 4)
            for idx in sorted_indices
        }
        
        return {
            "top_predictions": top_predictions,
            "top1_class": top_predictions[0]["class_name"],
            "top1_confidence": top_predictions[0]["confidence"],
            "all_probabilities": all_probabilities,
            "inference_time_ms": inference_time_ms,
            "tensor": tensor,
            "pil_image": pil_img
        }

if __name__ == "__main__":
    print("=" * 70)
    print("KIỂM THỬ MODULE DỰ ĐOÁN THỊ GIÁC (FISH_DISEASE_PREDICTOR)")
    print("=" * 70)
    predictor = FishDiseasePredictor()
    print("Mô hình khởi tạo thành công trên thiết bị:", predictor.device)
    print("Danh mục 4 lớp nhận diện:", predictor.idx_to_class)
    
    # Thử nghiệm với một ảnh mẫu từ tập dữ liệu
    test_img = Path("data/cache_224/test.jpg")
    if test_img.exists():
        res = predictor.predict(test_img, top_k=3)
        print(f"\nKết quả dự đoán ảnh {test_img.name} (Độ trễ: {res['inference_time_ms']} ms):")
        for p in res["top_predictions"]:
            print(f"  Top {p['rank']}: {p['class_name']:<35} | {p['confidence_percent']}")
    else:
        print("Không tìm thấy ảnh data/cache_224/test.jpg để chạy thử.")
