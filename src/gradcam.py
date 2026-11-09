# -*- coding: utf-8 -*-
import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
from pathlib import Path
from typing import Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

class FishGradCAM:
    """
    Bộ tạo bản đồ nhiệt Grad-CAM phục vụ giải thích chẩn đoán AI cho ảnh cá bệnh.
    """
    
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        """
        Khởi tạo Grad-CAM với mô hình và lớp tích chập mục tiêu.
        Mặc định sử dụng lớp model.features[-1] cho EfficientNet-B0.
        """
        self.model = model
        if target_layer is None:
            # Lớp tích chập cuối cùng của EfficientNet-B0
            self.target_layer = self.model.features[-1]
        else:
            self.target_layer = target_layer
            
        self.gradients = None
        self.activations = None
        
        # Đăng ký Forward và Backward Hooks
        self.hook_fwd = self.target_layer.register_forward_hook(self._forward_hook)
        self.hook_bwd = self.target_layer.register_full_backward_hook(self._backward_hook)
        
    def _forward_hook(self, module, input_t, output_t):
        self.activations = output_t.detach()
        
    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()
        
    def generate_cam(self, 
                     input_tensor: torch.Tensor, 
                     target_class: Optional[int] = None) -> Tuple[np.ndarray, int, float]:
        """
        Tính toán ma trận trọng số Grad-CAM và trả về mask chuẩn hóa [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1).detach().cpu().numpy()[0]
        
        if target_class is None:
            target_class = int(output.argmax(dim=1).item())
            
        target_score = output[0, target_class]
        target_score.backward()
        
        # Global Average Pooling (GAP) trên gradients
        # Shape gradients: [1, C, H, W] -> weights: [1, C, 1, 1]
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Tổ hợp tuyến tính giữa weights và feature map activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        
        # Resize về kích thước ảnh chuẩn (224, 224)
        cam = cv2.resize(cam, (224, 224))
        
        # Chuẩn hóa ma trận kích hoạt về miền [0.0, 1.0]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        confidence = float(probs[target_class])
        return cam, target_class, confidence
        
    def overlay_cam(self, 
                    pil_image: Image.Image, 
                    cam_mask: np.ndarray, 
                    alpha: float = 0.5, 
                    colormap: int = cv2.COLORMAP_JET) -> Tuple[np.ndarray, Image.Image]:
        """
        Tạo ảnh lớp phủ nhiệt (Heatmap Overlay) trên ảnh gốc.
        """
        # Đưa ảnh gốc về kích thước 224x224
        img_resized = pil_image.resize((224, 224), Image.Resampling.BILINEAR)
        img_np = np.array(img_resized)
        
        # Tạo bản đồ nhiệt màu từ mask [0, 1]
        heatmap = (cam_mask * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, colormap)
        # OpenCV applyColorMap trả về BGR -> chuyển sang RGB
        heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        # Trộn trọng số: alpha * heatmap + (1 - alpha) * original_image
        overlay = (alpha * heatmap_rgb + (1 - alpha) * img_np).astype(np.uint8)
        pil_overlay = Image.fromarray(overlay)
        return overlay, pil_overlay
        
    def save_visualization(self, 
                           pil_image: Image.Image, 
                           cam_mask: np.ndarray, 
                           output_path: Union[str, Path], 
                           title: Optional[str] = None,
                           predicted_class: Optional[str] = None, 
                           confidence: Optional[float] = None) -> str:
        """
        Lưu biểu đồ 3 khung hình đối chiếu trực quan (Ảnh gốc, Bản đồ nhiệt, Lớp phủ Grad-CAM).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        img_resized = pil_image.resize((224, 224))
        _, pil_overlay = self.overlay_cam(img_resized, cam_mask, alpha=0.5)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. Ảnh gốc
        axes[0].imshow(img_resized)
        axes[0].set_title("1. Ảnh Quan Sát Đầu Vào", fontsize=12, fontweight="bold")
        axes[0].axis("off")
        
        # 2. Bản đồ nhiệt Grad-CAM độc lập
        im2 = axes[1].imshow(cam_mask, cmap="jet", vmin=0, vmax=1)
        axes[1].set_title("2. Bản Đồ Kích Hoạt Vùng Bệnh", fontsize=12, fontweight="bold")
        axes[1].axis("off")
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        # 3. Lớp phủ Grad-CAM trên ảnh cá
        axes[2].imshow(pil_overlay)
        class_text = f": {predicted_class}" if predicted_class else ""
        conf_text = f" ({confidence:.1%})" if confidence is not None else ""
        axes[2].set_title(f"3. Lớp Phủ Giải Thích XAI{class_text}{conf_text}", fontsize=12, fontweight="bold")
        axes[2].axis("off")
        
        main_title = title or "Giải thích Chẩn đoán Bệnh Cá với Grad-CAM (EfficientNet-B0)"
        plt.suptitle(main_title, fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
        plt.close()
        return str(output_path)
        
    def remove_hooks(self):
        """Hủy đăng ký hooks để giải phóng bộ nhớ"""
        if self.hook_fwd:
            self.hook_fwd.remove()
        if self.hook_bwd:
            self.hook_bwd.remove()

if __name__ == "__main__":
    print("=" * 70)
    print("KIỂM THỬ MODULE GIẢI THÍCH GRAD-CAM (FISH_GRAD_CAM)")
    print("=" * 70)
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.predictor import FishDiseasePredictor
    
    predictor = FishDiseasePredictor()
    grad_cam = FishGradCAM(predictor.model)
    
    test_img = Path("data/cache_224/test.jpg")
    if test_img.exists():
        pred_res = predictor.predict(test_img)
        top1 = pred_res["top_predictions"][0]
        
        cam_mask, pred_cls, conf = grad_cam.generate_cam(pred_res["tensor"], target_class=top1["class_idx"])
        out_plot = Path("reports/sample_pipeline_gradcam.png")
        saved_path = grad_cam.save_visualization(
            pred_res["pil_image"],
            cam_mask,
            out_plot,
            predicted_class=top1["class_name"],
            confidence=conf
        )
        print(f"Đã tạo và lưu thành công bản đồ nhiệt Grad-CAM tại: {saved_path}")
    else:
        print("Không tìm thấy ảnh data/cache_224/test.jpg.")
