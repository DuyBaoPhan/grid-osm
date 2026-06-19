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

def split_crop_into_lines(crop_img: np.ndarray) -> List[np.ndarray]:
    """
    Phân tích hình ảnh crop của POI và cắt thành các dòng chữ đơn lẻ (nếu có nhiều dòng).
    Trả về danh sách các sub-image tương ứng với từng dòng chữ.
    """
    h_sz, w_sz = crop_img.shape[:2]
    if h_sz < 16:
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
                    if end_y - start_y >= 6:
                        bands.append((start_y, end_y))
                    in_band = False
                    
        if in_band:
            if h_sz - start_y >= 6:
                bands.append((start_y, h_sz))
                
        if len(bands) <= 1:
            return [crop_img]
            
        line_crops = []
        for sy, ey in bands:
            y1 = max(0, sy - 2)
            y2 = min(h_sz, ey + 2)
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
        
    # 2. Định dạng chuỗi số dài (mã số thuế, số điện thoại, zip code...)
    # Nếu dòng chứa chủ yếu là chữ số (ví dụ: trên 65% ký tự là số và có ít nhất 4 số)
    # Loại trừ trường hợp địa chỉ (ví dụ: "123 Lê Lợi" có chữ cái chiếm đa số)
    digits = sum(c.isdigit() for c in s)
    letters = sum(c.isalpha() for c in s)
    total = len(s.replace(" ", "").replace("-", "").replace(".", "").replace("/", ""))
    if total > 0 and digits >= 4 and (digits / total) > 0.65:
        if letters == 0 or (letters / total) < 0.20:
            return True

    return False

def _recognize_text_crop_vietocr(cv_img: np.ndarray, bbox: List[float], icon_side: str = "left") -> str:
    predictor = _get_vietocr_predictor()
    if predictor is None or cv_img is None:
        return ""

    h_img, w_img = cv_img.shape[:2]
    x1_orig, y1_orig, x2_orig, y2_orig = map(int, bbox)
    
    # 1. Dò tìm ranh giới icon trên crop gốc chưa pad để chính xác tuyệt đối
    original_box_crop = cv_img[max(0, y1_orig):min(h_img, y2_orig), max(0, x1_orig):min(w_img, x2_orig)]
    offset = _find_icon_boundary(original_box_crop, icon_side)
    
    # 2. Enlarge bbox (Padding 10px theo code USER)
    pad = 10
    x1 = max(0, x1_orig - pad)
    y1 = max(0, y1_orig - pad)
    x2 = min(w_img, x2_orig + pad)
    y2 = min(h_img, y2_orig + pad)

    crop = cv_img[y1:y2, x1:x2]
    if crop.size == 0:
        return ""

    # 3. Loại bỏ phần biểu tượng (icon) khỏi crop đã pad sử dụng offset đã dịch chuyển
    try:
        # Tính toán offset trong hệ tọa độ của crop đã pad
        # Cạnh trái dịch chuyển sang trái x1_orig - x1 pixel (chính là pad, trừ phi chạm biên ảnh)
        shift_x = x1_orig - x1
        shift_y = y1_orig - y1
        
        if icon_side == "left":
            crop = crop[:, max(0, offset + shift_x - 2):]
        elif icon_side == "right":
            crop = crop[:, :max(1, (x2_orig - x1) - offset + 2)]
        elif icon_side == "top":
            crop = crop[max(0, offset + shift_y - 2):, :]
    except Exception as e:
        logger.debug("Lỗi loại bỏ icon trước OCR: %s", e)

    try:
        # Tách crop thành các dòng chữ đơn lẻ
        line_crops = split_crop_into_lines(crop)
        
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
        icon_size = int(height)
        icon_size_top = min(int(height), int(width))
        
        h_img, w_img = img.shape[:2]
        y1_sq, y2_sq = max(0, int(top)), min(h_img, int(bottom))
        x1_l, x2_l = max(0, int(left)), min(w_img, int(left + icon_size))
        x1_r, x2_r = max(0, int(right - icon_size)), min(w_img, int(right))
        
        # Top square: lấy vùng vuông ở giữa-trên của bbox
        cx_center = (left + right) / 2.0
        x1_t = max(0, int(cx_center - icon_size_top / 2))
        x2_t = min(w_img, int(cx_center + icon_size_top / 2))
        y1_t = max(0, int(top))
        y2_t = min(h_img, int(top + icon_size_top))
        
        left_sq = img[y1_sq:y2_sq, x1_l:x2_l]
        right_sq = img[y1_sq:y2_sq, x1_r:x2_r]
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
        if score_top > score_left + THRESHOLD and score_top > score_right + THRESHOLD:
            icon_side = "top"
            cx = (left + right) / 2.0
        elif score_right > score_left + THRESHOLD:
            icon_side = "right"
            cx = right - height / 2.0
        else:
            icon_side = "left"
            cx = left + height / 2.0
            
        cy = top + height / 2.0
        
        # Mở rộng bbox xuống dưới để ôm trọn dòng chữ thứ 2 (hoặc 3)
        left_exp, top_exp, right_exp, bottom_exp = expand_bbox_downward(img, [left, top, right, bottom], max_expand=28)
        b_expanded = [left_exp, top_exp, right_exp, bottom_exp]
        
        # OCR text dùng logic phóng to 6x và phân tích đa dòng (loại bỏ icon)
        text = _recognize_text_crop_vietocr(img, b_expanded, icon_side)
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
        }
        pois.append(poi_item)
 
        if SAVE_POI_CROPS:
            # Làm sạch tên file để loại bỏ ký tự không hợp lệ trên Windows
            safe_name = re.sub(r'[\\/*?:"<>|]', "", name).replace(" ", "_").strip()
            if not safe_name:
                safe_name = f"Unknown_{i}"
            filename = f"tile_{tx}_{ty}_poi_{i}_{safe_name}.png"
            output_path = os.path.join(POI_CROPS_DIR, filename)
            save_poi_crop(screenshot_bytes, poi_item, output_path)
 
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

def _find_icon_boundary(crop_img: np.ndarray, icon_side: str) -> int:
    """
    Tìm số pixel cần bỏ từ cạnh icon-side để đến vùng text.

    Nguyên tắc an toàn:
    - Ngưỡng rất chặt (SAT=100, FG=0.25) để chỉ phát hiện icon thực sự (màu sắc rực rỡ)
      không bị nhiễu bởi tile bản đồ (noise lác đác) hay chữ màu xanh/cam (sat ~40-70)
    - Kết quả luôn được clamp vào [MIN_PX, MAX_PX] để không bao giờ cắt quá mức
    - Nếu không phát hiện icon rõ ràng, fallback về DEFAULT_PX (20px) - bảo thủ, an toàn
    """
    DEFAULT_PX = 22   # fallback: cắt 22px (icon ~20px, kết hợp gap ~4px)
    MIN_PX     = 20   # giữ lại tối thiểu 20px để sạch icon
    MAX_PX     = 24   # cắt tối đa 24px để không phạm vào chữ (kể cả chữ có màu như UBND)

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
        VAL_MAX      = 230  # loại nền trắng
        FG_RATIO     = 0.25 # ≥25% pixel trong cột phải là icon-colored
        MAX_SCAN     = MAX_PX + 4
        MARGIN       = 2
        REQUIRED_GAP = 5

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


def save_poi_crop(image_bytes: bytes, poi: dict, output_path: str, crop_x: int = 0, crop_y: int = 0):
    """Lưu ảnh crop của POI, chỉ giữ phần text (bỏ phần icon)."""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return
        
        bbox = poi.get("bbox")
        if not bbox:
            return
        
        # bbox có dạng [left, top, width, height]
        l, t, w, h = map(int, bbox)
        h_img, w_img = img.shape[:2]

        icon_side = poi.get("icon_side", "left")
        pad = 5

        # Quét pixel để tìm ranh giới icon. Fallback cố định 20px (bảo thủ, không cắt vào text)
        bbox_crop = img[max(0, t):min(h_img, t + h), max(0, l):min(w_img, l + w)]
        offset = _find_icon_boundary(bbox_crop, icon_side)

        if icon_side == "top":
            x1 = max(0, l - pad)
            y1 = max(0, t + offset - 2)
            x2 = min(w_img, l + w + pad)
            y2 = min(h_img, t + h + pad)
            if y2 - y1 < 8:  # fallback
                y1 = max(0, t - pad)
                y2 = min(h_img, t + h + pad)
        elif icon_side == "right":
            x1 = max(0, l - pad)
            y1 = max(0, t - pad)
            x2 = min(w_img, l + w - offset + 2)
            y2 = min(h_img, t + h + pad)
            if x2 - x1 < 8:  # fallback
                x1 = max(0, l - pad)
                x2 = min(w_img, l + w + pad)
        else:  # left
            x1 = max(0, l + offset - 2)
            y1 = max(0, t - pad)
            x2 = min(w_img, l + w + pad)
            y2 = min(h_img, t + h + pad)
            if x2 - x1 < 8:  # fallback
                x1 = max(0, l - pad)
                x2 = min(w_img, l + w + pad)
        
        crop = img[y1:y2, x1:x2]
        if crop.size > 0:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, crop)
            logger.info("  [Crop saved] %s", os.path.basename(output_path))
    except Exception as e:
        logger.warning("Không thể lưu ảnh crop POI: %s", e)
