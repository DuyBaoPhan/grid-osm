"""
Test script: chạy YOLO + VietOCR trực tiếp trên ảnh screenshot đã lưu
để kiểm tra xem model có detect được không.
"""
import cv2
import os
import sys
from PIL import Image

# Patch torch.load for PyTorch 2.6 compatibility before importing ultralytics/yolo
try:
    import torch
    _orig_load = torch.load
    def _patched_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _orig_load(*args, **kwargs)
    torch.load = _patched_load
except ImportError:
    pass

# Add project root to path
sys.path.insert(0, r"D:\grid-osm")
sys.path.insert(0, r"D:\grid-osm\src")

# Tìm ảnh screenshot đầu tiên
screenshot_dir = r"D:\grid-osm\screenshots"
img_files = [f for f in os.listdir(screenshot_dir) if f.endswith(".png") and "tile_" in f]
if not img_files:
    print("Không tìm thấy ảnh nào trong thư mục screenshots!")
    sys.exit(1)

# Dùng tile_0_0.png hoặc file đầu tiên tìm được
target = "tile_0_0.png" if "tile_0_0.png" in img_files else img_files[0]
img_path = os.path.join(screenshot_dir, target)
print(f"[TEST] Dùng ảnh: {img_path}")

img = cv2.imread(img_path)
print(f"[TEST] Kích thước ảnh: {img.shape}")  # (h, w, c)

# =========================
# Load YOLO
# =========================
from ultralytics import YOLO
model_path = r"D:\grid-osm\model\best.pt"
print(f"[TEST] Load YOLO từ: {model_path}")
model = YOLO(model_path)

# Test với conf thấp để xem có box nào không
for conf_test in [0.1, 0.2, 0.3, 0.5]:
    results = model.predict(img_path, conf=conf_test, save=False, verbose=False)
    num_boxes = len(results[0].boxes) if results and results[0].boxes else 0
    print(f"  conf={conf_test} → {num_boxes} boxes")

# Lấy kết quả conf=0.2 để vẽ
results = model.predict(img_path, conf=0.2, save=False, verbose=True)
print(f"\n[TEST] Tổng boxes với conf=0.2: {len(results[0].boxes)}")

# =========================
# Load VietOCR
# =========================
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
config = Cfg.load_config_from_name("vgg_transformer")
config["device"] = "cpu"
config["predictor"]["beamsearch"] = True
ocr = Predictor(config)

# =========================
# OCR + vẽ box
# =========================
for i, box in enumerate(results[0].boxes.xyxy):
    x1, y1, x2, y2 = map(int, box.cpu().numpy())
    pad = 10
    x1p = max(0, x1 - pad)
    y1p = max(0, y1 - pad)
    x2p = min(img.shape[1], x2 + pad)
    y2p = min(img.shape[0], y2 + pad)
    
    crop = img[y1p:y2p, x1p:x2p]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    pil_img = Image.fromarray(gray)
    text = ocr.predict(pil_img)
    print(f"  Box {i+1}: [{x1},{y1},{x2},{y2}] → '{text}'")
    
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(img, text, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Vẽ điểm tâm và tọa độ pixel màu vàng (BGR: 0, 255, 255)
    height = y2 - y1
    icon_size = int(height)
    y1_sq, y2_sq = max(0, int(y1)), min(img.shape[0], int(y2))
    x1_l, x2_l = max(0, int(x1)), min(img.shape[1], int(x1 + icon_size))
    x1_r, x2_r = max(0, int(x2 - icon_size)), min(img.shape[1], int(x2))
    
    left_sq = img[y1_sq:y2_sq, x1_l:x2_l]
    right_sq = img[y1_sq:y2_sq, x1_r:x2_r]
    
    def get_square_score(sq):
        import numpy as np
        if sq is None or sq.size == 0:
            return -999.0
        try:
            hsv = cv2.cvtColor(sq, cv2.COLOR_BGR2HSV)
            s = hsv[:, :, 1]
            v = hsv[:, :, 2]
            
            # Foreground mask: not (Value > 215 and Saturation < 30)
            fg_mask = ~((v > 215) & (s < 30))
            
            # Center 50% region foreground density
            h_sz, w_sz = sq.shape[:2]
            cy_min, cy_max = int(0.25 * h_sz), int(0.75 * h_sz)
            cx_min, cx_max = int(0.25 * w_sz), int(0.75 * w_sz)
            center_mask = fg_mask[cy_min:cy_max, cx_min:cx_max]
            center_fg_ratio = np.mean(center_mask) if center_mask.size > 0 else 0.0
            
            mean_sat = float(np.mean(s))
            mean_val = float(np.mean(v))
            
            # Combined score: favors saturated colors, dark center region, penalizes background brightness
            score = mean_sat + 60.0 * center_fg_ratio - 0.2 * mean_val
            return score
        except Exception:
            return -999.0
            
    score_left = get_square_score(left_sq)
    score_right = get_square_score(right_sq)
    
    if score_right > score_left + 10.0:
        cx = int(x2 - height / 2.0)
    else:
        cx = int(x1 + height / 2.0)
        
    cy = int(y1 + height / 2.0)
    print(f"    [DIAG] text: '{text}', score_left={score_left:.1f}, score_right={score_right:.1f}, x1={x1}, x2={x2}, cx={cx}")
    cv2.circle(img, (cx, cy), 6, (0, 255, 255), -1)
    cv2.circle(img, (cx, cy), 7, (0, 0, 0), 1)
    cv2.putText(img, f"({cx}, {cy})", (cx + 10, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

# Lưu kết quả ra file
out_path = os.path.join(screenshot_dir, "TEST_RESULT.png")
cv2.imwrite(out_path, img)
print(f"\n[TEST] Đã lưu kết quả vào: {out_path}")
