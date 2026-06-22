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

# =========================
# Load YOLO
# =========================
from ultralytics import YOLO
model_path = r"D:\grid-osm\model\best.pt"
print(f"[TEST] Load YOLO từ: {model_path}")
model = YOLO(model_path)

# Duyệt qua tất cả các file để test
for target in sorted(img_files):
    if target == "TEST_RESULT.png" or "TEST_RESULT_" in target:
        continue
    img_path = os.path.join(screenshot_dir, target)
    print(f"\n==================================================")
    print(f"[TEST] Dùng ảnh: {img_path}")
    
    img = cv2.imread(img_path)
    if img is None:
        continue
    print(f"[TEST] Kích thước ảnh: {img.shape}")  # (h, w, c)
    
    # Lấy kết quả conf=0.2 để vẽ
    results = model.predict(img_path, conf=0.2, save=False, verbose=False)
    print(f"[TEST] Tổng boxes với conf=0.2: {len(results[0].boxes)}")
    
    # =========================
    # OCR + vẽ box sử dụng module vision của dự án
    # =========================
    import vision
    
    for i, box in enumerate(results[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box.cpu().numpy())
        
        # Xác định vị trí icon để truyền vào vision
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
                
                fg_mask = ~((v > 215) & (s < 30))
                
                h_sz, w_sz = sq.shape[:2]
                cy_min, cy_max = int(0.25 * h_sz), int(0.75 * h_sz)
                cx_min, cx_max = int(0.25 * w_sz), int(0.75 * w_sz)
                center_mask = fg_mask[cy_min:cy_max, cx_min:cx_max]
                center_fg_ratio = np.mean(center_mask) if center_mask.size > 0 else 0.0
                
                mean_sat = float(np.mean(s))
                mean_val = float(np.mean(v))
                
                score = mean_sat + 60.0 * center_fg_ratio - 0.2 * mean_val
                return score
            except Exception:
                return -999.0
                
        score_left = get_square_score(left_sq)
        score_right = get_square_score(right_sq)
        
        # Xác định chính xác tâm icon và icon_side như vision.py
        score_top = -999.0
        THRESHOLD = 10.0
        
        if max(score_left, score_right) < THRESHOLD:
            icon_side = "left"
        elif score_right > score_left + THRESHOLD:
            icon_side = "right"
        else:
            icon_side = "left"
            
        icon_w_search = int(30 * 1.0)
        icon_h_search = int(30 * 1.0)
        h_img, w_img = img.shape[:2]
        
        if icon_side == "left":
            x1_s = max(0, int(x1))
            x2_s = min(w_img, int(x1 + icon_w_search))
            y1_s = max(0, int(y1))
            y2_s = min(h_img, int(y1 + icon_h_search))
        elif icon_side == "right":
            x1_s = max(0, int(x2 - icon_w_search))
            x2_s = min(w_img, int(x2))
            y1_s = max(0, int(y1))
            y2_s = min(h_img, int(y1 + icon_h_search))
        else:
            x1_s = y1_s = x2_s = y2_s = 0
    
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if x2_s > x1_s and y2_s > y1_s:
            search_area = img[y1_s:y2_s, x1_s:x2_s]
            import numpy as np
            hsv = cv2.cvtColor(search_area, cv2.COLOR_BGR2HSV)
            s = hsv[:, :, 1]
            v = hsv[:, :, 2]
            
            SAT_THRESH = 35
            VAL_MIN = 60
            VAL_MAX = 256
            
            icon_mask = (s > SAT_THRESH) & (v > VAL_MIN) & (v < VAL_MAX)
            pixel_count = np.sum(icon_mask)
            
            if pixel_count >= int(15 * 1.0):
                ys, xs = np.where(icon_mask)
                cx = int(x1_s + np.mean(xs))
                cy = int(y1_s + np.mean(ys))
            else:
                cx = int(x1_s + (x2_s - x1_s) / 2.0)
                cy = int(y1_s + (y2_s - y1_s) / 2.0)
        
        # Gọi hàm xử lý OCR chuẩn của vision.py
        text = vision._recognize_text_crop_vietocr(img, [x1, y1, x2, y2], icon_side=icon_side, scale=1.0, cx=cx, cy=cy)
        print(f"  Box {i+1}: [{x1},{y1},{x2},{y2}] → '{text}'")
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(img, text, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.circle(img, (cx, cy), 6, (0, 255, 255), -1)
        cv2.circle(img, (cx, cy), 7, (0, 0, 0), 1)
        cv2.putText(img, f"({cx}, {cy})", (cx + 10, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    
    # Lưu kết quả ra file
    out_path = os.path.join(screenshot_dir, f"TEST_RESULT_{target}")
    cv2.imwrite(out_path, img)
    print(f"[TEST] Đã lưu kết quả vào: {out_path}")
