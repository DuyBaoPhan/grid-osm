# =============================================================
# geometry.py — bbox/text area helpers extracted from original vision.py
# =============================================================

import logging
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

def expand_bbox_downward(img: np.ndarray, bbox: List[float], max_expand: int = 28) -> List[float]:
    """
    Tự động quét và mở rộng bbox xuống dưới để ôm trọn dòng chữ thứ 2 (hoặc 3) của POI.
    Bằng cách phát hiện ranh giới bong bóng (bubble) nền của nhãn Google Maps.
    """
    h_img, w_img = img.shape[:2]
    left, top, right, bottom = bbox
    
    L = max(0, int(left))
    R = min(w_img, int(right))
    T = max(0, int(top))
    B = min(h_img, int(bottom))
    
    if R - L <= 0 or B - T <= 0:
        return bbox
        
    try:
        # 1. Trích xuất vùng ảnh hiện tại để tìm màu nền chủ đạo (median)
        box_crop = img[T:B, L:R]
        hsv_crop = cv2.cvtColor(box_crop, cv2.COLOR_BGR2HSV)
        
        bg_h = np.median(hsv_crop[:, :, 0])
        bg_s = np.median(hsv_crop[:, :, 1])
        bg_v = np.median(hsv_crop[:, :, 2])
        
        is_white_bg = (bg_s < 25)
        
        new_bottom = B
        
        # 2. Quét từng dòng bên dưới cạnh bottom hiện tại
        for y in range(B, min(h_img, B + max_expand)):
            row_pixels = img[y, L:R]
            row_hsv = cv2.cvtColor(row_pixels.reshape(1, -1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
            
            matching_count = 0
            for i in range(len(row_hsv)):
                h, s, v = row_hsv[i]
                
                # Check similarity to background
                is_bg = False
                if is_white_bg:
                    if s < 35 and abs(int(v) - int(bg_v)) < 30:
                        is_bg = True
                else:
                    dh = min(abs(int(h) - int(bg_h)), 180 - abs(int(h) - int(bg_h)))
                    if dh < 15 and abs(int(s) - int(bg_s)) < 40 and abs(int(v) - int(bg_v)) < 40:
                        is_bg = True
                        
                # Check if it is text/foreground (contrast)
                is_text = False
                if bg_v > 150:
                    if v < 110:
                        is_text = True
                else:
                    if v > 170:
                        is_text = True
                        
                if is_bg or is_text:
                    matching_count += 1
                    
            ratio = matching_count / len(row_hsv) if len(row_hsv) > 0 else 0.0
            
            # Ngưỡng nghiêm ngặt hơn (0.78) để không lan vào mặt đường/nền bản đồ màu xám/trắng
            if ratio > 0.78:
                new_bottom = y
            else:
                break
                
        # Thêm 3px padding an toàn để tránh cắt mất đuôi chữ (g, y, p, q...)
        final_bottom = min(h_img, new_bottom + 3)
        
        # Co lại (trim) để vừa vặn với dòng chữ cuối cùng (bỏ trống phần trắng dư thừa ở dưới bong bóng)
        last_text_y = B
        for y in range(B, int(final_bottom)):
            row_pixels = img[y, L:R]
            row_hsv = cv2.cvtColor(row_pixels.reshape(1, -1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
            text_pixel_count = 0
            for i in range(len(row_hsv)):
                h, s, v = row_hsv[i]
                is_text = False
                if bg_v > 150:
                    if v < 110:
                        is_text = True
                else:
                    if v > 170:
                        is_text = True
                if is_text:
                    text_pixel_count += 1
            
            width_row = R - L
            if text_pixel_count >= max(2, int(0.015 * width_row)):
                last_text_y = y
                
        # Giữ lại khoảng cách an toàn 4px dưới dòng chữ cuối cùng
        trimmed_bottom = min(final_bottom, last_text_y + 4)
        return [left, top, right, float(trimmed_bottom)]
        
    except Exception as e:
        logger.warning("Lỗi trong expand_bbox_downward: %s", e)
        return bbox


def detect_text_area(
    crop_img: np.ndarray,
    scale: float = 1.0
) -> Tuple[int, int, int, int]:
    """
    Sử dụng OpenCV contours được nới lỏng để định vị vùng chữ trong ảnh crop POI.
    Tự động lọc bỏ biểu tượng ở biên (trái/phải) bằng giải thuật hình học.
    Trả về tọa độ chữ cục bộ trong crop_img: (x1, y1, x2, y2)
    """
    h_crop, w_crop = crop_img.shape[:2]
    if h_crop < 4 or w_crop < 4:
        return 0, 0, w_crop, h_crop

    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    bg_val = np.median(gray)
    if bg_val > 127:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    char_boxes = []
    h_min = max(3, int(4 * scale))
    w_min = max(1, int(1 * scale))
    area_min = max(2, int(4 * scale * scale))
    
    for ctr in contours:
        x, y, w, h = cv2.boundingRect(ctr)
        area = cv2.contourArea(ctr)
        aspect_ratio = w / float(h) if h > 0 else 0
        # Relaxed filters: no h_max, w_max, area_max, aspect_ratio <= 4.0 limits
        if (h >= h_min) and (w >= w_min):
            if (area >= area_min) and (aspect_ratio >= 0.05):
                char_boxes.append((x, y, w, h))
                
    if not char_boxes:
        return 0, 0, w_crop, h_crop

    # Connected components to group character boxes into lines/words
    n = len(char_boxes)
    parent = list(range(n))
    
    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]
        
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    for i in range(n):
        x1, y1, w1, h1 = char_boxes[i]
        for j in range(i + 1, n):
            x2, y2, w2, h2 = char_boxes[j]
            
            if x1 <= x2:
                dist_x = x2 - (x1 + w1)
            else:
                dist_x = x1 - (x2 + w2)
                
            overlap_y = min(y1 + h1, y2 + h2) - max(y1, y2)
            min_h = min(h1, h2)
            has_vertical_overlap = (overlap_y / float(min_h)) > 0.3 if min_h > 0 else False
            
            if has_vertical_overlap and dist_x < int(28 * scale):
                union(i, j)
                
    components = {}
    for i in range(n):
        root = find(i)
        if root not in components:
            components[root] = []
        components[root].append(char_boxes[i])
        
    valid_components = []
    for root, idxs in components.items():
        # Sắp xếp các contour theo chiều ngang từ trái qua phải
        sorted_boxes = sorted(idxs, key=lambda b: b[0])
        
        # 1. Kiểm tra và loại bỏ icon ở bên trái (nếu có)
        pop_thresh = int(18 * scale)
        if len(sorted_boxes) >= 3:
            b_first = sorted_boxes[0]
            b_second = sorted_boxes[1]
            gap_left = b_second[0] - (b_first[0] + b_first[2])
            # Nếu contour đầu tiên to/rộng cả chiều ngang lẫn dọc (>= 18px) và có khoảng trống với chữ
            if b_first[2] >= pop_thresh and b_first[3] >= pop_thresh and gap_left >= int(3 * scale):
                sorted_boxes.pop(0)
        
        # 2. Kiểm tra và loại bỏ icon ở bên phải (nếu có)
        if len(sorted_boxes) >= 3:
            b_last = sorted_boxes[-1]
            b_prev = sorted_boxes[-2]
            gap_right = b_last[0] - (b_prev[0] + b_prev[2])
            if b_last[2] >= pop_thresh and b_last[3] >= pop_thresh and gap_right >= int(3 * scale):
                sorted_boxes.pop()
        
        # Nếu sau khi loại bỏ icon vẫn còn ít nhất 1 hộp hợp lệ (đáp ứng từ ngắn hoặc dính nét)
        if len(sorted_boxes) >= 1:
            c_xmin = min(b[0] for b in sorted_boxes)
            c_xmax = max(b[0] + b[2] for b in sorted_boxes)
            c_ymin = min(b[1] for b in sorted_boxes)
            c_ymax = max(b[1] + b[3] for b in sorted_boxes)
            c_w = c_xmax - c_xmin
            c_h = c_ymax - c_ymin
            
            valid_components.append({
                "xmin": c_xmin,
                "xmax": c_xmax,
                "ymin": c_ymin,
                "ymax": c_ymax,
                "w": c_w,
                "h": c_h,
                "cy": (c_ymin + c_ymax) / 2.0,
                "boxes": sorted_boxes
            })
            
    if not valid_components:
        x_min = min(b[0] for b in char_boxes)
        y_min = min(b[1] for b in char_boxes)
        x_max = max(b[0] + b[2] for b in char_boxes)
        y_max = max(b[1] + b[3] for b in char_boxes)
    else:
        # Sắp xếp các dòng theo chiều dọc y từ trên xuống
        valid_components.sort(key=lambda c: c["cy"])
        
        # Tìm anchor component gần tâm dọc nhất của crop (nhãn chính)
        ref_y = h_crop / 2.0
        anchor_idx = min(range(len(valid_components)), key=lambda idx: abs(valid_components[idx]["cy"] - ref_y))
        
        # Lan truyền lên trên và xuống dưới để nhận các dòng chữ liên tục có khoảng cách nhỏ
        kept_components = [valid_components[anchor_idx]]
        max_gap = int(12 * scale)
        
        # Đi lên trên từ anchor
        curr_idx = anchor_idx
        while curr_idx > 0:
            above = valid_components[curr_idx - 1]
            curr = valid_components[curr_idx]
            gap = curr["ymin"] - above["ymax"]
            if gap <= max_gap:
                kept_components.insert(0, above)
                curr_idx -= 1
            else:
                break
                
        # Đi xuống dưới từ anchor
        curr_idx = anchor_idx
        while curr_idx < len(valid_components) - 1:
            below = valid_components[curr_idx + 1]
            curr = valid_components[curr_idx]
            gap = below["ymin"] - curr["ymax"]
            if gap <= max_gap:
                kept_components.append(below)
                curr_idx += 1
            else:
                break
                
        x_min = min(c["xmin"] for c in kept_components)
        y_min = min(c["ymin"] for c in kept_components)
        x_max = max(c["xmax"] for c in kept_components)
        y_max = max(c["ymax"] for c in kept_components)

    pad_px = int(4 * scale)
    tx1 = max(0, x_min - pad_px)
    ty1 = max(0, y_min - pad_px)
    tx2 = min(w_crop, x_max + pad_px)
    ty2 = min(h_crop, y_max + pad_px)
    
    return tx1, ty1, tx2, ty2
