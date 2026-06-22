# =============================================================
# vision.py — Nhận diện POI bằng YOLOv8 & VietOCR (Code chuẩn của USER)
# =============================================================

import asyncio
import io
import logging
import os
import re
from typing import List, Tuple, Optional

import cv2
import numpy as np
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

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from config import (
    YOLO_MODEL_PATH,
    VIETOCR_MODEL,
    VIETOCR_DEVICE,
    SAVE_POI_CROPS,
    POI_CROPS_DIR,
)

logger = logging.getLogger(__name__)

_YOLO_MODEL = None
_VIETOCR_PREDICTOR = None

def _get_yolo_model():
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        if YOLO is None:
            logger.error("Thư viện 'ultralytics' chưa được cài đặt.")
            return None
        try:
            _YOLO_MODEL = YOLO(YOLO_MODEL_PATH)
            logger.info("Loaded YOLO model from %s", YOLO_MODEL_PATH)
        except Exception as e:
            logger.error("Không thể load YOLO model: %s", e)
    return _YOLO_MODEL

def _get_vietocr_predictor():
    global _VIETOCR_PREDICTOR
    if _VIETOCR_PREDICTOR is not None:
        return _VIETOCR_PREDICTOR
    try:
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor
        config = Cfg.load_config_from_name(VIETOCR_MODEL)
        config["device"] = VIETOCR_DEVICE
        config["predictor"]["beamsearch"] = True
        # Theo code của USER: cnn pretrained = False
        if "cnn" in config:
            config["cnn"]["pretrained"] = False
        _VIETOCR_PREDICTOR = Predictor(config)
        logger.info("Loaded VietOCR model=%s", VIETOCR_MODEL)
        return _VIETOCR_PREDICTOR
    except Exception as exc:
        logger.warning("Không load được VietOCR: %s", exc)
        return None

def _strip_vietnamese_accents(s: str) -> str:
    """Loại bỏ hoàn toàn dấu tiếng Việt và đưa về chữ thường (Dùng cho deduplicate)."""
    s = s.lower()
    s = re.sub(r"[àáảãạăằắẳẵặâầấẩẫậ]", "a", s)
    s = re.sub(r"[èéẻẽẹêềếểễệ]", "e", s)
    s = re.sub(r"[ìíỉĩị]", "i", s)
    s = re.sub(r"[òóỏõọôồốổỗộơờớởỡợ]", "o", s)
    s = re.sub(r"[ùúủũụưừứửữự]", "u", s)
    s = re.sub(r"[ỳýỷỹỵ]", "y", s)
    s = re.sub(r"[đ]", "d", s)
    return s

def _clean_spelling(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    return text

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

def split_crop_into_lines(crop_img: np.ndarray, scale: float = 1.0) -> List[np.ndarray]:
    """
    Phân tích hình ảnh crop của POI và cắt thành các dòng chữ đơn lẻ (nếu có nhiều dòng).
    Trả về danh sách các sub-image tương ứng với từng dòng chữ.
    """
    h_sz, w_sz = crop_img.shape[:2]
    if h_sz < int(16 * scale):
        return [crop_img]
        
    try:
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        
        bg_s = np.median(s)
        bg_v = np.median(v)
        is_white_bg = (bg_s < 25)
        
        row_bg_ratios = []
        for y in range(h_sz):
            row_s = s[y, :]
            row_v = v[y, :]
            
            if is_white_bg:
                bg_pixels = np.sum((row_s < 35) & (row_v > 200))
            else:
                bg_pixels = np.sum((np.abs(row_s.astype(int) - int(bg_s)) < 40) & (np.abs(row_v.astype(int) - int(bg_v)) < 40))
                
            row_bg_ratios.append(bg_pixels / w_sz)
            
        row_bg_ratios = np.array(row_bg_ratios)
        
        # Tìm khoảng trống giữa các dòng (tỷ lệ nền > 93%)
        GAP_THRESH = 0.93
        is_gap = row_bg_ratios > GAP_THRESH
        
        bands = []
        in_band = False
        start_y = 0
        
        for y in range(h_sz):
            if not is_gap[y]:
                if not in_band:
                    start_y = y
                    in_band = True
            else:
                if in_band:
                    end_y = y
                    if end_y - start_y >= int(6 * scale):
                        bands.append((start_y, end_y))
                    in_band = False
                    
        if in_band:
            if h_sz - start_y >= int(6 * scale):
                bands.append((start_y, h_sz))
                
        if len(bands) <= 1:
            return [crop_img]
            
        line_crops = []
        for sy, ey in bands:
            y1 = max(0, sy - int(2 * scale))
            y2 = min(h_sz, ey + int(2 * scale))
            line_crops.append(crop_img[y1:y2, :])
            
        return line_crops
        
    except Exception as e:
        logger.debug("Lỗi khi chia dòng OCR: %s", e)
        return [crop_img]

def _is_junk_line(s: str) -> bool:
    """Loại bỏ các dòng chữ rác không phải là tên địa điểm (số điện thoại, mã số thuế, đánh giá...)."""
    s = s.strip()
    if not s:
        return True
    
    # 1. Định dạng rating dạng "4.74 (53)" hoặc "(53)" hoặc "4.7"
    if re.match(r'^\d+(\.\d+)?\s*\(\d+\)$', s) or re.match(r'^\(\d+\)$', s):
        return True
    if re.match(r'^\d+(\.\d+)?$', s):
        return True
        
    # Mới: Loại bỏ các dòng rating kèm chữ phụ như "4.6% (35) - Đặt bàn ngay tại Hum"
    if re.search(r'^\d+(\.\d+)?%\s*\(\d+\)', s) or re.search(r'^\d+(\.\d+)?\s*\(\d+\)\s*-\s*', s):
        return True
        
    # Check for consecutive digits on the original string (with spaces/punctuation)
    # - 5 consecutive digits anywhere (e.g. "00110", "18000000...")
    # - or 3 consecutive zeros at the start (e.g. "000Trang...")
    if re.search(r'\d{5,}', s) or re.match(r'^0{3,}', s):
        return True
        
    # Remove spaces and common punctuation to analyze characters
    clean_s = re.sub(r'[^a-zA-Z0-9]', '', s)
    if not clean_s:
        return True
        
    digits = sum(c.isdigit() for c in clean_s)
    letters = sum(c.isalpha() for c in clean_s)
    total = len(clean_s)
    
    # 2. Định dạng chuỗi số dài (mã số thuế, số điện thoại, zip code...)
    # Nếu dòng chứa nhiều chữ số (trên 65% ký tự và có ít nhất 4 số)
    # Loại trừ trường hợp địa chỉ (ví dụ: "123 Lê Lợi" có chữ cái chiếm đa số)
    if total > 0 and digits >= 4 and (digits / total) > 0.65:
        if letters == 0 or (letters / total) < 0.20:
            return True

    # 3. Các từ viết tắt nhiễu do quét nhầm viền icon Google Maps (ví dụ: Cons, COLInters, commes...)
    # Nhận diện các từ rác bắt đầu bằng tiền tố và kết thúc bằng hậu tố đặc trưng của nét tròn
    junk_pattern = (
        r'(?i)^(con|col|com|comm|cor|cot|cos|can|trn|trans|contra|contr|inter|hype|como|trns|colinter|ant|antr|dis|disc|discol)'
        r'(s|es|ers|ins|inters|ess|he|ste|cars|c|ar|shone|gers|tracess|ces|cess|monums|phousness|anshone|cars|ication|che|tess|ness|mogers'
        r'|anaTis|naTis|ceptor|interpLat|interpOUm|prOUm|OUm|cept|conceptor|com|intersm|intersM|vers|sstering|teris|tracOM|cOM|COM)?$'
    )
    if re.match(junk_pattern, clean_s):
        return True
    # Từ rác standalone không khớp pattern prefix+suffix
    junk_standalone = {'quantousus', 'qUantousus', 'unstitute', 'discepter', 'quantous', 'collo', 'coliner', 'communs', 'co1', 'derrigermin1'}
    if clean_s.lower() in junk_standalone:
        return True
        
    return False

def _clean_junk_words(s: str) -> str:
    words = s.split()
    cleaned_words = []
    for w in words:
        w_clean = re.sub(r'[^a-zA-Z0-9]', '', w)
        if re.search(r'\d{4,}', w_clean) or re.match(r'^0{3,}', w_clean):
            continue
        junk_pattern = (
            r'(?i)^(con|col|com|comm|cor|cot|cos|can|trn|trans|contra|contr|inter|hype|como|trns|colinter|disterat|terat|ant|antr|dis|disc|discol)'
            r'(s|es|ers|ins|inters|ess|he|ste|cars|c|ar|shone|gers|tracess|ces|cess|monums|phousness|anshone|cars|ication|che|tess|ness|mogers'
            r'|eritonerizede|anaTis|naTis|ceptor|interpLat|interpOUm|prOUm|OUm|cept|conceptor|com|intersm|intersM|vers|sstering|teris|tracOM|cOM|COM)+$'
        )
        if re.match(junk_pattern, w_clean):
            continue
        junk_standalone = {'quantousus', 'unstitute', 'discepter', 'quantous', 'collo', 'coliner', 'communs', 'co1', 'derrigermin1'}
        if w_clean.lower() in junk_standalone:
            continue
        cleaned_words.append(w)
    
    res = " ".join(cleaned_words).strip()
    res = re.sub(r'^[\s,\-/]+|[\s,\-/]+$', '', res).strip()
    return res

def detect_text_area(
    crop_img: np.ndarray,
    scale: float = 1.0,
    cy_local: Optional[float] = None,
    icon_side: str = "left"
) -> Tuple[int, int, int, int]:
    """
    Sử dụng OpenCV contours để định vị vùng chữ trong ảnh crop POI.
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
    h_max = int(18 * scale)
    w_min = max(1, int(1 * scale))
    w_max = int(24 * scale)
    area_min = max(2, int(4 * scale * scale))
    area_max = int(250 * scale * scale)
    
    for ctr in contours:
        x, y, w, h = cv2.boundingRect(ctr)
        area = cv2.contourArea(ctr)
        aspect_ratio = w / float(h) if h > 0 else 0
        if (h_min <= h <= h_max) and (w_min <= w <= w_max):
            if (area_min <= area <= area_max) and (0.05 <= aspect_ratio <= 4.0):
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
        cy1 = y1 + h1 / 2.0
        for j in range(i + 1, n):
            x2, y2, w2, h2 = char_boxes[j]
            cy2 = y2 + h2 / 2.0
            
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
        c_xmin = min(b[0] for b in idxs)
        c_xmax = max(b[0] + b[2] for b in idxs)
        c_ymin = min(b[1] for b in idxs)
        c_ymax = max(b[1] + b[3] for b in idxs)
        c_w = c_xmax - c_xmin
        c_h = c_ymax - c_ymin
        c_count = len(idxs)
        if c_count >= 3 and c_w >= int(25 * scale):
            valid_components.append({
                "xmin": c_xmin,
                "xmax": c_xmax,
                "ymin": c_ymin,
                "ymax": c_ymax,
                "w": c_w,
                "h": c_h,
                "cy": (c_ymin + c_ymax) / 2.0,
                "boxes": idxs
            })

    # Lọc bỏ các component (dòng chữ) thuộc về địa điểm khác dựa vào cy_local (tọa độ y của icon trong crop)
    if cy_local is not None and valid_components:
        filtered_components = []
        if icon_side in ("left", "right"):
            # Đối với icon nằm ngang hàng với dòng 1:
            # Dòng 1 nằm ở tầm cy_local. Dòng 2, 3 nằm dưới đó.
            # Dòng của địa điểm khác thường cách xa cy_local (đỉnh dòng dưới ymin > cy_local + 42*scale)
            # Dòng phía trên nếu bị quét nhầm có ymax < cy_local - 15*scale
            min_y = cy_local - 15 * scale
            max_y = cy_local + 42 * scale
        else:
            # icon_side == "top": icon ở trên cùng, text ở dưới icon.
            # Dòng của địa điểm khác ở dưới có ymin > cy_local + 58*scale
            # Dòng phía trên nếu bị quét nhầm có ymax < cy_local - 8*scale
            min_y = cy_local - 8 * scale
            max_y = cy_local + 58 * scale
            
        for c in valid_components:
            # c["ymin"] là đỉnh của dòng, c["ymax"] là đáy của dòng
            if c["ymax"] >= min_y and c["ymin"] <= max_y:
                filtered_components.append(c)
        
        # Chỉ áp dụng nếu sau khi lọc vẫn còn ít nhất 1 dòng, tránh bị rỗng hoàn toàn
        if filtered_components:
            valid_components = filtered_components
            
    if not valid_components:
        x_min = min(b[0] for b in char_boxes)
        y_min = min(b[1] for b in char_boxes)
        x_max = max(b[0] + b[2] for b in char_boxes)
        y_max = max(b[1] + b[3] for b in char_boxes)
    else:
        # Sắp xếp các dòng theo chiều dọc y từ trên xuống
        valid_components.sort(key=lambda c: c["cy"])
        
        # Tìm anchor component gần vị trí icon dọc nhất (hoặc tâm dọc của crop nếu không có cy_local)
        ref_y = cy_local if cy_local is not None else (h_crop / 2.0)
        anchor_idx = min(range(len(valid_components)), key=lambda idx: abs(valid_components[idx]["cy"] - ref_y))
        
        # Lan truyền lên trên và xuống dưới để nhận các dòng chữ liên tục có khoảng cách nhỏ
        kept_components = [valid_components[anchor_idx]]
        max_gap = int(12 * scale)  # Khoảng cách dòng tối đa cho phép trong cùng 1 POI
        
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

def _recognize_text_crop_vietocr(cv_img: np.ndarray, bbox: List[float], icon_side: str = "left", scale: float = 1.0, cx: float = None, cy: float = None) -> str:
    predictor = _get_vietocr_predictor()
    if predictor is None or cv_img is None:
        return ""

    h_img, w_img = cv_img.shape[:2]
    x1_orig, y1_orig, x2_orig, y2_orig = map(int, bbox)
    
    pad = int(10 * scale)
    y1 = max(0, y1_orig - pad)
    y2 = min(h_img, y2_orig + pad)
    x1 = max(0, x1_orig - pad)
    x2 = min(w_img, x2_orig + pad)
    
    raw_crop = cv_img[y1:y2, x1:x2]
    if raw_crop.size == 0:
        return ""

    cy_local = None
    if cy is not None:
        cy_local = cy - y1
    tx1, ty1, tx2, ty2 = detect_text_area(raw_crop, scale, cy_local=cy_local, icon_side=icon_side)
    crop = raw_crop[ty1:ty2, tx1:tx2]
    if crop.size == 0:
        return ""

    try:
        # Tách crop thành các dòng chữ đơn lẻ
        line_crops = split_crop_into_lines(crop, scale)
        
        texts = []
        for line_crop in line_crops:
            if line_crop.size == 0:
                continue
                
            # Tiền xử lý mới: Đổi từ BGR sang RGB, phóng to 2x (thay vì 6x), và không dùng GaussianBlur
            rgb_line = cv2.cvtColor(line_crop, cv2.COLOR_BGR2RGB)
            rgb_2x = cv2.resize(
                rgb_line,
                None,
                fx=2,
                fy=2,
                interpolation=cv2.INTER_CUBIC
            )
            
            # Predict trực tiếp trên ảnh màu sắc nét
            pil_img = Image.fromarray(rgb_2x)
            text = predictor.predict(pil_img)
            text_str = (text or "").strip()
            if text_str:
                # Clean rating prefix first (ví dụ: "4.74 (53) - Nhà hàng" -> "Nhà hàng")
                text_clean = re.sub(r'^\d+(\.\d+)?\s*\(\d+\)\s*(-\s*)?', '', text_str).strip()
                text_clean = _clean_junk_words(text_clean)
                if text_clean and not _is_junk_line(text_clean):
                    texts.append(text_clean)
                
        if not texts:
            return ""
        return " / ".join(texts)
    except Exception as e:
        logger.debug("VietOCR error: %s", e)
        return ""

async def extract_pois_from_screenshot(
    screenshot_bytes: bytes,
    tx: int = 0,
    ty: int = 0,
    img_metadata: dict = None,
) -> Tuple[List[dict], bool]:
    model = _get_yolo_model()
    if model is None:
        return [], False

    nparr = np.frombuffer(screenshot_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return [], False

    # Tự động tính toán scale factor (mật độ điểm ảnh)
    scale = 1.0
    if img_metadata and "scale" in img_metadata:
        scale = float(img_metadata["scale"])
    else:
        # Ước tính từ độ phân giải ảnh (1080px làm tiêu chuẩn)
        from config import SCREENSHOT_H
        scale = img.shape[0] / SCREENSHOT_H

    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, lambda: model.predict(img, conf=0.2))
    
    pois = []
    if not results or len(results) == 0:
        return [], False

    result = results[0]
    boxes = result.boxes
    
    for i, box in enumerate(boxes):
        b = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        
        # Xác định biểu tượng (icon) nằm ở trái / phải / trên của bounding box
        left = float(b[0])
        top = float(b[1])
        right = float(b[2])
        bottom = float(b[3])
        height = bottom - top
        width = right - left
        
        # Crop squares for scores using a fixed icon size of 32 * scale
        icon_size_check = int(32 * scale)
        h_img, w_img = img.shape[:2]
        
        # Left square
        y1_l = max(0, int(top))
        y2_l = min(h_img, int(top + icon_size_check))
        x1_l = max(0, int(left))
        x2_l = min(w_img, int(left + icon_size_check))
        
        # Right square
        y1_r = max(0, int(top))
        y2_r = min(h_img, int(top + icon_size_check))
        x1_r = max(0, int(right - icon_size_check))
        x2_r = min(w_img, int(right))
        
        # Top square: lấy vùng vuông ở giữa-trên của bbox
        cx_center = (left + right) / 2.0
        x1_t = max(0, int(cx_center - icon_size_check / 2))
        x2_t = min(w_img, int(cx_center + icon_size_check / 2))
        y1_t = max(0, int(top))
        y2_t = min(h_img, int(top + icon_size_check))
        
        left_sq = img[y1_l:y2_l, x1_l:x2_l]
        right_sq = img[y1_r:y2_r, x1_r:x2_r]
        top_sq = img[y1_t:y2_t, x1_t:x2_t]
        
        def get_square_score(sq):
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
        score_top = get_square_score(top_sq)
        
        THRESHOLD = 10.0
        # Nếu không có icon rõ ràng ở bất kỳ phía nào → mặc định là left (hầu hết POI đều có icon bên trái)
        # để bắt được cả các icon màu xám/nhạt không vượt qua THRESHOLD
        if max(score_left, score_right, score_top) < THRESHOLD:
            icon_side = "left"
        elif score_top > score_left + THRESHOLD and score_top > score_right + THRESHOLD:
            icon_side = "top"
        elif score_right > score_left + THRESHOLD:
            icon_side = "right"
        else:
            icon_side = "left"

        # Định vị chính xác tâm icon bằng phương pháp tìm trọng tâm (centroid) màu sắc
        icon_w_search = int(30 * scale)
        icon_h_search = int(30 * scale)
        h_img, w_img = img.shape[:2]
        
        if icon_side == "left":
            x1_s = max(0, int(left))
            x2_s = min(w_img, int(left + icon_w_search))
            y1_s = max(0, int(top))
            y2_s = min(h_img, int(top + icon_h_search))
        elif icon_side == "right":
            x1_s = max(0, int(right - icon_w_search))
            x2_s = min(w_img, int(right))
            y1_s = max(0, int(top))
            y2_s = min(h_img, int(top + icon_h_search))
        elif icon_side == "top":
            cx_center = (left + right) / 2.0
            x1_s = max(0, int(cx_center - icon_w_search / 2))
            x2_s = min(w_img, int(cx_center + icon_w_search / 2))
            y1_s = max(0, int(top))
            y2_s = min(h_img, int(top + icon_h_search))
        else:
            x1_s = y1_s = x2_s = y2_s = 0

        cx, cy = (left + right) / 2.0, (top + bottom) / 2.0
        if x2_s > x1_s and y2_s > y1_s:
            search_area = img[y1_s:y2_s, x1_s:x2_s]
            hsv = cv2.cvtColor(search_area, cv2.COLOR_BGR2HSV)
            s = hsv[:, :, 1]
            v = hsv[:, :, 2]
            
            # Ngưỡng màu để lọc ra biểu tượng có màu sắc của Google Maps
            SAT_THRESH = 35
            VAL_MIN = 60
            VAL_MAX = 256
            
            icon_mask = (s > SAT_THRESH) & (v > VAL_MIN) & (v < VAL_MAX)
            pixel_count = np.sum(icon_mask)
            
            if pixel_count >= int(15 * scale * scale):
                ys, xs = np.where(icon_mask)
                cx = x1_s + np.mean(xs)
                cy = y1_s + np.mean(ys)
            else:
                # Fallback nếu không có đủ pixel màu (icon xám)
                cx = x1_s + (x2_s - x1_s) / 2.0
                cy = y1_s + (y2_s - y1_s) / 2.0
        
        # Mở rộng bbox xuống dưới để ôm trọn dòng chữ thứ 2 (hoặc 3)
        left_exp, top_exp, right_exp, bottom_exp = expand_bbox_downward(img, [left, top, right, bottom], max_expand=int(28 * scale))
        b_expanded = [left_exp, top_exp, right_exp, bottom_exp]
        
        # OCR text dùng logic phóng to 6x và phân tích đa dòng (loại bỏ icon)
        text = _recognize_text_crop_vietocr(img, b_expanded, icon_side, scale, cx=cx, cy=cy)
        name = _clean_spelling(text)
        
        if not name:
            name = f"Unknown_{i}"
  
        poi_item = {
            "name": name,
            "x": cx,
            "y": cy,
            "confidence": conf,
            "bbox": [float(left_exp), float(top_exp), float(right_exp - left_exp), float(bottom_exp - top_exp)],
            "has_icon": True,
            "icon_side": icon_side,
            "yolo_height": float(height),
        }
        pois.append(poi_item)
  
        if SAVE_POI_CROPS:
            # Làm sạch tên file để loại bỏ ký tự không hợp lệ trên Windows
            safe_name = re.sub(r'[\\/*?:"<>|]', "", name).replace(" ", "_").strip()
            if not safe_name:
                safe_name = f"Unknown_{i}"
            filename = f"tile_{tx}_{ty}_poi_{i}_{safe_name}.png"
            output_path = os.path.join(POI_CROPS_DIR, filename)
            save_poi_crop(screenshot_bytes, poi_item, output_path, scale)
    return pois, False

def draw_detections(image_bytes: bytes, pois: List[dict]) -> bytes:
    """Vẽ box đỏ, text và tọa độ pixel (x, y) lên ảnh (theo logic debug của USER)."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return image_bytes

    for p in pois:
        bbox = p.get("bbox")
        name = p.get("name", "")
        if not bbox:
            continue
        l, t, w, h = map(int, bbox)
        # Vẽ box ĐỎ (BGR: 0, 0, 255)
        cv2.rectangle(img, (l, t), (l + w, t + h), (0, 0, 255), 3)
        # Bỏ vẽ tên XANH (BGR: 255, 0, 0) để tránh rối mắt theo yêu cầu
        # cv2.putText(img, name, (l, max(t - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Vẽ điểm tâm và tọa độ pixel màu vàng (BGR: 0, 255, 255)
        cx = p.get("x")
        cy = p.get("y")
        if cx is not None and cy is not None:
            cx_i, cy_i = int(cx), int(cy)
            cv2.circle(img, (cx_i, cy_i), 6, (0, 255, 255), -1)  # Circle
            cv2.circle(img, (cx_i, cy_i), 7, (0, 0, 0), 1)       # Outline for contrast
            coord_str = f"({cx_i}, {cy_i})"
            cv2.putText(img, coord_str, (cx_i + 10, cy_i + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()

def enhance_for_detection(image_bytes: bytes) -> bytes:
    return image_bytes

def remove_background(image_bytes: bytes) -> bytes:
    return image_bytes

def _find_icon_boundary(crop_img: np.ndarray, icon_side: str, scale: float = 1.0) -> int:
    """
    Tìm số pixel cần bỏ từ cạnh icon-side để đến vùng text.

    Nguyên tắc an toàn:
    - Ngưỡng thấp (SAT=35, FG=0.25) để nhận diện được cả icon màu nhạt
    - Kết quả luôn được clamp vào [MIN_PX, MAX_PX] để không bao giờ cắt quá mức
    - Nếu không phát hiện icon rõ ràng, fallback về DEFAULT_PX - bảo thủ, an toàn
    - Các giá trị tĩnh theo scale (đã xác nhận qua thực nghiệm) thay vì dùng yolo_height
      (yolo_height quá lớn ~40-60px, gây cắt lấn vào chữ tên địa điểm)
    """
    DEFAULT_PX = int(32 * scale)   # icon ~20px + gap ~5px + viền ~7px
    MIN_PX     = int(18 * scale)   # tối thiểu: icon ~15px nhỏ nhất
    MAX_PX     = int(42 * scale)   # tối đa: tránh lấn vào chữ

    if crop_img is None or crop_img.size == 0:
        return DEFAULT_PX
    try:
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        h_sz, w_sz = crop_img.shape[:2]

        # Ngưỡng thấp hơn (35) để nhận diện được các icon màu nhạt/desaturated (như UBND, xám, xanh nhạt)
        SAT_THRESH   = 35  # icon sat > 120-200; chữ màu xanh sat ~40-70 → bị loại
        VAL_MIN      = 60   # loại shadow
        VAL_MAX      = 256  # cho phép bắt tất cả các màu icon sáng chói (V=255)
        FG_RATIO     = 0.25 # ≥25% pixel trong cột phải là icon-colored
        MAX_SCAN     = MAX_PX + int(4 * scale)
        MARGIN       = int(2 * scale)
        REQUIRED_GAP = int(10 * scale)

        icon_mask = (s > SAT_THRESH) & (v > VAL_MIN) & (v < VAL_MAX)

        # Pre-check: nếu toàn bộ crop có nhiều pixel màu sắc (nền cam/xanh của label quảng cáo),
        # không thể phát hiện icon đơn lẻ → dùng DEFAULT_PX để không cắt nhầm text
        overall_ratio = float(np.sum(icon_mask)) / (h_sz * w_sz) if (h_sz * w_sz > 0) else 0.0
        if overall_ratio > 0.40:
            return DEFAULT_PX

        def _scan(indices, is_col: bool) -> int:
            last_icon = -1
            gap = 0
            found = False
            for idx in indices:
                col_data = icon_mask[:, idx] if is_col else icon_mask[idx, :]
                denom = h_sz if is_col else w_sz
                ratio = float(np.sum(col_data)) / denom
                if ratio > FG_RATIO:
                    last_icon = idx
                    gap = 0
                    found = True
                elif found:
                    gap += 1
                    if gap >= REQUIRED_GAP:
                        break
            return last_icon

        result = None
        if icon_side == "left":
            last = _scan(range(min(w_sz, MAX_SCAN)), is_col=True)
            if last >= 0:
                result = last + MARGIN
        elif icon_side == "right":
            last = _scan(range(w_sz - 1, max(-1, w_sz - MAX_SCAN - 1), -1), is_col=True)
            if last >= 0:
                result = w_sz - last + MARGIN
        elif icon_side == "top":
            last = _scan(range(min(h_sz, MAX_SCAN)), is_col=False)
            if last >= 0:
                result = last + MARGIN

        if result is not None:
            # Clamp kết quả vào [MIN_PX, MAX_PX] → không bao giờ cắt quá mức
            return max(MIN_PX, min(MAX_PX, result))


    except Exception:
        pass
    return DEFAULT_PX


def save_poi_crop(image_bytes: bytes, poi: dict, output_path: str, scale: float = 1.0):
    """Lưu ảnh crop của POI, chỉ giữ phần text (bỏ phần icon)."""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return
        
        bbox = poi.get("bbox")
        if not bbox:
            return
        
        l, t, w, h = map(int, bbox)
        h_img, w_img = img.shape[:2]

        pad = int(10 * scale)
        y1 = max(0, t - pad)
        y2 = min(h_img, t + h + pad)
        x1 = max(0, l - pad)
        x2 = min(w_img, l + w + pad)

        raw_crop = img[y1:y2, x1:x2]
        if raw_crop.size == 0:
            return

        cy_local = None
        cy_global = poi.get("y")
        if cy_global is not None:
            cy_local = cy_global - y1
        icon_side = poi.get("icon_side", "left")

        tx1, ty1, tx2, ty2 = detect_text_area(raw_crop, scale, cy_local=cy_local, icon_side=icon_side)
        crop = raw_crop[ty1:ty2, tx1:tx2]
        if crop.size > 0:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, crop)
            logger.info("  [Crop saved] %s", os.path.basename(output_path))
    except Exception as e:
        logger.warning("Không thể lưu ảnh crop POI: %s", e)
