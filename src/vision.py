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
    PADDLE_TEXT_DET_ENABLED,
)

logger = logging.getLogger(__name__)

_YOLO_MODEL = None
_VIETOCR_PREDICTOR = None
_PADDLE_TEXT_DETECTOR = None

def _get_paddle_text_detector():
    """Lazy-load PaddleOCR để chỉ detect/recognize text."""
    global _PADDLE_TEXT_DETECTOR
    if not PADDLE_TEXT_DET_ENABLED:
        return None
    if _PADDLE_TEXT_DETECTOR is False:
        return None
    if _PADDLE_TEXT_DETECTOR is not None:
        return _PADDLE_TEXT_DETECTOR
    try:
        os.environ.setdefault("FLAGS_use_mkldnn", "0")
        os.environ.setdefault("FLAGS_enable_pir_api", "0")
        from paddleocr import PaddleOCR
        # Không truyền det/rec/cls ở constructor: một số bản PaddleOCR báo "Unknown argument: det".
        _PADDLE_TEXT_DETECTOR = PaddleOCR(
            use_angle_cls=False,
            lang="en",
        )
        logger.info("Loaded PaddleOCR detector")
        return _PADDLE_TEXT_DETECTOR
    except Exception as exc:
        _PADDLE_TEXT_DETECTOR = False
        logger.warning("Không load được PaddleOCR: %s", exc)
        return None

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

def _normalize_vietnamese_place_phrases(text: str) -> str:
    """Chuẩn hóa cụm địa danh Việt bằng gazetteer ngoài file và fuzzy guard."""
    try:
        from src.vietnam_places import normalize_place_phrases
    except ImportError:
        try:
            from vietnam_places import normalize_place_phrases
        except ImportError:
            return text
    return normalize_place_phrases(text)


def _remove_adjacent_duplicate_ocr_tokens(text: str) -> str:
    """Xóa token OCR lặp liền kề như `Bình Bình`, không đụng brand ALLCAPS."""
    if not text:
        return text

    matches = list(re.finditer(r'[A-Za-zÀ-ỹĐđ0-9]+', text))
    if len(matches) < 2:
        return text

    remove_indexes = set()
    prev_key = None
    prev_token = None
    for idx, match in enumerate(matches):
        token = match.group(0)
        key = _strip_vietnamese_accents(token).lower()
        if (
            prev_key == key
            and len(key) >= 3
            and prev_token is not None
            and not prev_token.isupper()
            and not token.isupper()
            and not any(ch.isdigit() for ch in prev_token + token)
        ):
            remove_indexes.add(idx)
            continue
        prev_key = key
        prev_token = token

    if not remove_indexes:
        return text

    out = []
    last = 0
    for idx, match in enumerate(matches):
        if idx in remove_indexes:
            out.append(text[last:match.start()].rstrip())
            last = match.end()
            continue
        out.append(text[last:match.start()])
        out.append(match.group(0))
        last = match.end()
    out.append(text[last:])
    return re.sub(r'\s+', ' ', ''.join(out)).strip()


def _clean_spelling(text: str) -> str:
    # Loại bỏ các ký tự dấu nháy kép, nháy đơn, backtick, gạch chéo ngược và ngoặc rác từ icon/viền crop
    text = re.sub(r"[\"\'`\\\[\]\{\}]", "", text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = _normalize_vietnamese_place_phrases(text)
    # OCR hay nhầm Cúng -> Cũng trong ngữ cảnh dịch vụ/đồ cúng; chỉ sửa khi có cụm ngữ cảnh rõ.
    text = re.sub(r'(?i)\b(Đồ|Dịch vụ Đồ)\s+Cũng\b', lambda m: m.group(1) + ' Cúng', text)
    text = re.sub(r'(?i)\bCũng\s+Trọn\s+Gói\b', 'Cúng Trọn Gói', text)
    text = _remove_adjacent_duplicate_ocr_tokens(text)
    return text


def _has_vietnamese_mark(token: str) -> bool:
    """True nếu token có dấu tiếng Việt, gồm cả chữ đ/Đ."""
    return bool(re.search(r'[À-ỹĐđ]', token or ""))


def _merge_primary_diacritics(primary_text: str, selected_text: str) -> str:
    """
    Khi selected/normalized đúng cấu trúc hơn nhưng sai dấu nhẹ, mượn dấu từ Primary.
    Chỉ thay token nếu:
    - token bỏ dấu giống nhau
    - cả Primary và selected đều có dấu Việt
    - không đổi số/ký tự brand không dấu
    Ví dụ: Thợ -> Thọ sẽ khôi phục Thợ; Xe không bị đổi thành Xẻ.
    """
    primary_tokens = re.findall(r'[A-Za-zÀ-ỹĐđ0-9]+', primary_text or "")
    selected_tokens = re.findall(r'[A-Za-zÀ-ỹĐđ0-9]+', selected_text or "")
    if not primary_tokens or len(primary_tokens) != len(selected_tokens):
        return selected_text

    replacements = []
    changed = False
    for p_tok, s_tok in zip(primary_tokens, selected_tokens):
        if p_tok == s_tok:
            replacements.append(s_tok)
            continue
        if (
            _strip_vietnamese_accents(p_tok) == _strip_vietnamese_accents(s_tok)
            and _has_vietnamese_mark(p_tok)
            and _has_vietnamese_mark(s_tok)
            and not any(ch.isdigit() for ch in p_tok + s_tok)
        ):
            replacements.append(p_tok)
            changed = True
        else:
            replacements.append(s_tok)

    if not changed:
        return selected_text

    repl_iter = iter(replacements)
    return re.sub(r'[A-Za-zÀ-ỹĐđ0-9]+', lambda _: next(repl_iter), selected_text)


def _vietnamese_tone_preference(token: str) -> int:
    """Heuristic nhẹ để chọn dấu Việt tự nhiên hơn giữa hai token cùng base."""
    t = token or ""
    score = 0
    # Dấu sắc/huyền/nặng thường ổn định hơn hỏi/ngã trong OCR nhỏ; không ép tuyệt đối.
    score += 2 * len(re.findall(r'[áéíóúýấắếốớứ]', t, flags=re.IGNORECASE))
    score += 1 * len(re.findall(r'[àèìòùỳầằềồờừ]', t, flags=re.IGNORECASE))
    score += 1 * len(re.findall(r'[ạẹịọụỵậặệộợự]', t, flags=re.IGNORECASE))
    score -= 1 * len(re.findall(r'[ảẻỉỏủỷẩẳểổởử]', t, flags=re.IGNORECASE))
    score -= 1 * len(re.findall(r'[ãẽĩõũỹẫẵễỗỡữ]', t, flags=re.IGNORECASE))
    return score


def _has_vietnamese_shaped_vowel(token: str) -> bool:
    """True nếu token có nguyên âm Việt đặc thù dễ bị OCR làm mất: ơ/ư/ă/â/ê/ô."""
    return bool(re.search(r'[ăằắẳẵặâầấẩẫậêềếểễệôồốổỗộơờớởỡợưừứửữự]', token or "", flags=re.IGNORECASE))


def _merge_best_diacritics(primary_text: str, selected_text: str) -> str:
    """Chọn dấu tốt hơn giữa Primary và selected khi token cùng base bỏ dấu."""
    primary_tokens = re.findall(r'[A-Za-zÀ-ỹĐđ0-9]+', primary_text or "")
    selected_tokens = re.findall(r'[A-Za-zÀ-ỹĐđ0-9]+', selected_text or "")
    if not primary_tokens or len(primary_tokens) != len(selected_tokens):
        return selected_text

    replacements = []
    changed = False
    for p_tok, s_tok in zip(primary_tokens, selected_tokens):
        if p_tok == s_tok:
            replacements.append(s_tok)
            continue
        if (
            _strip_vietnamese_accents(p_tok) == _strip_vietnamese_accents(s_tok)
            and _has_vietnamese_mark(p_tok)
            and _has_vietnamese_mark(s_tok)
            and not any(ch.isdigit() for ch in p_tok + s_tok)
        ):
            if _has_vietnamese_shaped_vowel(p_tok) and not _has_vietnamese_shaped_vowel(s_tok):
                replacements.append(p_tok)
                changed = True
            elif _vietnamese_tone_preference(p_tok) > _vietnamese_tone_preference(s_tok):
                replacements.append(p_tok)
                changed = True
            else:
                replacements.append(s_tok)
        else:
            replacements.append(s_tok)

    if not changed:
        return selected_text
    repl_iter = iter(replacements)
    return re.sub(r'[A-Za-zÀ-ỹĐđ0-9]+', lambda _: next(repl_iter), selected_text)

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
        pad_y = max(4, int(4 * scale))
        for sy, ey in bands:
            y1 = max(0, sy - pad_y)
            y2 = min(h_sz, ey + pad_y)
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
        r'|anaTis|naTis|interpLat|interpOUm|prOUm|OUm|com|intersm|intersM|vers|sstering|teris|tracOM|cOM|COM)?$'
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
            r'|eritonerizede|anaTis|naTis|interpLat|interpOUm|prOUm|OUm|com|intersm|intersM|vers|sstering|teris|tracOM|cOM|COM)+$'
        )
        if re.match(junk_pattern, w_clean):
            continue
        junk_standalone = {'quantousus', 'unstitute', 'discepter', 'quantous', 'collo', 'coliner', 'communs', 'co1', 'derrigermin1', 'disserian'}
        if w_clean.lower() in junk_standalone:
            continue
        cleaned_words.append(w)
    
    res = " ".join(cleaned_words).strip()
    res = re.sub(r'^[\s,\-/]+|[\s,\-/]+$', '', res).strip()
    return res


def _looks_like_junk_token(token: str, *, is_edge: bool = False) -> bool:
    """Nhận diện token OCR vô nghĩa để không xuất ra kết quả cuối."""
    raw = (token or "").strip()
    clean = re.sub(r'[^A-Za-zÀ-ỹĐđ0-9]', '', raw)
    if not clean:
        return True

    key = _strip_vietnamese_accents(clean).lower()
    if key in {"tp", "cn", "k", "q", "p"}:
        return False
    if any(ch.isdigit() for ch in clean):
        return False
    if re.search(r'[À-ỹĐđ]', clean):
        return False
    if clean.isupper() and 2 <= len(clean) <= 6:
        return False
    if is_edge and len(clean) <= 2 and not clean[:1].isupper():
        return True

    artifact_suffixes = (
        "erian", "iness", "teris", "cess", "tess", "tracess", "oum",
        "natis", "shone", "mogers", "phousness", "interpoum",
    )
    if len(key) >= 7 and key.endswith(artifact_suffixes):
        return True
    if key in {"disserian", "quantousus", "unstitute", "discepter"}:
        return True
    if len(key) >= 8 and not clean[:1].isupper():
        vowels = sum(ch in "aeiouy" for ch in key)
        letters = sum(ch.isalpha() for ch in key)
        if letters and vowels / letters < 0.25:
            return True
    return False


def _clean_final_ocr_text(text: str) -> str:
    """Cleanup cuối: không để ký tự/từ rác lọt ra output."""
    if not text:
        return ""
    text = _clean_spelling(text)
    parts = [p.strip() for p in re.split(r'\s*/\s*', text) if p.strip()]
    cleaned_parts = []
    for part_idx, part in enumerate(parts):
        part = re.sub(r'^[^A-Za-zÀ-ỹĐđ0-9]+|[^A-Za-zÀ-ỹĐđ0-9]+$', '', part).strip()
        part = _clean_junk_words(part)
        words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', part)
        kept_words = []
        for idx, word in enumerate(words):
            is_edge = idx == 0 or idx == len(words) - 1
            if _looks_like_junk_token(word, is_edge=is_edge):
                continue
            kept_words.append(word)
        if kept_words:
            # Nếu không xóa token nào, giữ nguyên dấu câu hợp lệ trong segment (vd: `Chợ Nga, TP`).
            if len(kept_words) == len(words):
                cleaned_parts.append(part)
            else:
                cleaned_parts.append(" ".join(kept_words))
    cleaned = " / ".join(cleaned_parts)
    cleaned = _clean_ocr_edge_segments(cleaned) if '_clean_ocr_edge_segments' in globals() else cleaned
    return cleaned.strip()


def _junk_token_count(text: str) -> int:
    parts = [p.strip() for p in re.split(r'\s*/\s*', text or "") if p.strip()]
    count = 0
    for part_idx, part in enumerate(parts):
        words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', part)
        for idx, word in enumerate(words):
            is_edge = idx == 0 or idx == len(words) - 1
            if _looks_like_junk_token(word, is_edge=is_edge):
                count += 1
    return count

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


def _looks_like_bad_ocr(text: str) -> bool:
    """Nhận diện kết quả OCR có khả năng rác để thử fallback PaddleOCR recognition."""
    s = (text or "").strip()
    if not s:
        return True
    clean = re.sub(r'[^A-Za-zÀ-ỹ0-9]', '', s)
    if len(clean) <= 4:
        return True
    lower = _strip_vietnamese_accents(s).lower()
    junk_tokens = (
        "obst", "obs", "overstress", "couth", "quts", "orns", "orng",
        "ongame", "oriem", "ducas", "seruper", "postotice", "pertume",
        "obtrined", "parigheness", "qutminh", "qut"
    )
    if any(tok in lower for tok in junk_tokens):
        return True
    words = re.findall(r'[A-Za-zÀ-ỹ0-9]+', s)
    if words and len(words) <= 2 and not any(ch.isdigit() for ch in s):
        vowel_count = sum(ch in 'aeiouyàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ' for ch in lower)
        letter_count = sum(ch.isalpha() for ch in lower)
        if letter_count and vowel_count / letter_count < 0.28:
            return True
    return False


def _score_ocr_text_quality(text: str, fx: float = 1.0) -> float:
    """Chấm điểm chất lượng OCR tổng quát, không phụ thuộc keyword/tên riêng."""
    s = (text or "").strip()
    if not s:
        return -9999

    words = re.findall(r'[A-Za-zÀ-ỹ0-9]+', s)
    letters = re.findall(r'[A-Za-zÀ-ỹ]', s)
    digits = re.findall(r'\d', s)
    vietnamese_marks = re.findall(r'[À-ỹ]', s)
    clean_len = len(re.sub(r'[^A-Za-zÀ-ỹ0-9]', '', s))

    score = clean_len
    score += 8 * max(0, s.count(' / '))
    score += 4 * max(0, len(words) - 1)
    score += 2 * len(vietnamese_marks)

    if _looks_like_bad_ocr(s):
        score -= 80

    if letters:
        digit_ratio = len(digits) / max(1, len(letters) + len(digits))
        if digit_ratio > 0.35:
            score -= int(60 * digit_ratio)

    artifact_penalty = _ocr_artifact_score(s) if '_ocr_artifact_score' in globals() else 0
    score -= 14 * artifact_penalty
    if '_junk_token_count' in globals():
        score -= 45 * _junk_token_count(s)

    # Token 1 ký tự ở đầu/cuối thường là mẩu icon hoặc chữ rác.
    if words and len(words[0]) == 1 and len(words) > 1:
        score -= 18
    if words and len(words[-1]) == 1 and len(words) > 1:
        score -= 12

    # Áp dụng cùng rule cho từng segment ngăn bởi '/', vì lỗi thường xuất hiện dạng "D Little...".
    for segment in re.split(r'\s*/\s*', s):
        seg_words = re.findall(r'[A-Za-zÀ-ỹ0-9]+', segment)
        if len(seg_words) > 1 and len(seg_words[0]) == 1:
            score -= 28
        if len(seg_words) > 1 and len(seg_words[-1]) == 1:
            score -= 16
        # Ending ngắn sau token dài thường là chữ bị cụt/nối dòng sai: "Steakhous / Ste".
        if len(seg_words) == 1 and len(seg_words[0]) <= 3 and segment == s.split('/')[-1].strip():
            previous_words = re.findall(r'[A-Za-zÀ-ỹ0-9]+', ' / '.join(re.split(r'\s*/\s*', s)[:-1]))
            if previous_words and len(previous_words[-1]) >= 7:
                score -= 35

    # Dòng quá ngắn chỉ chấp nhận nếu nó là label ngắn thật; cho điểm thấp để variant dài hơn thắng.
    if clean_len <= 4:
        score -= 30

    return score + 0.05 * fx


def _ocr_artifact_score(text: str) -> int:
    """Đếm dấu hiệu OCR méo chữ/dấu câu, không phụ thuộc tên riêng."""
    s = text or ""
    score = 0
    score += 3 * len(re.findall(r'[?#*"“”]', s))
    score += 2 * len(re.findall(r'[():;]', s))
    score += 2 * len(re.findall(r'(?<=\w)[\-–—](?=\w)', s))  # dấu gạch chen trong token: -laan
    score += 2 * len(re.findall(r'\d[A-Za-zÀ-ỹ]|[A-Za-zÀ-ỹ]\d', s))  # 5Chạt
    score += len(re.findall(r'[^\w\sÀ-ỹ/&.,%+\-–—]', s))
    for segment in re.split(r'\s*/\s*', s):
        words = re.findall(r'[A-Za-zÀ-ỹ0-9]+', segment)
        if len(words) > 1 and len(words[0]) == 1:
            score += 2
        if len(words) > 1 and len(words[-1]) == 1:
            score += 1
    return score


def _normalized_regresses_quality(primary_text: str, norm_text: str) -> bool:
    """Giữ Primary nếu normalized không cải thiện rõ mà làm méo token/dấu câu/chính tả."""
    if not primary_text or not norm_text or _looks_like_bad_ocr(primary_text):
        return False

    primary_artifacts = _ocr_artifact_score(primary_text)
    norm_artifacts = _ocr_artifact_score(norm_text)
    primary_tokens = _ocr_tokens(primary_text)
    norm_tokens = _ocr_tokens(norm_text)
    if not primary_tokens or not norm_tokens:
        return False

    overlap = len(set(primary_tokens) & set(norm_tokens))
    overlap_ratio = overlap / max(1, min(len(primary_tokens), len(norm_tokens)))
    token_delta = abs(len(norm_tokens) - len(primary_tokens))

    # Normalized cùng nội dung gần như Primary nhưng nhiều artifact hơn: giữ Primary.
    if overlap_ratio >= 0.70 and norm_artifacts > primary_artifacts:
        return True

    # Normalized không có artifact hơn, nhưng chỉ là biến thể chính tả/dấu câu của Primary.
    # Nếu Primary đã tốt, tránh override chỉ khi Normalized thực sự có thêm artifact.
    if overlap_ratio >= 0.78 and token_delta <= 2 and norm_artifacts > primary_artifacts:
        return True

    # Normalized thêm segment/từ ngoài khi primary đã đủ dài thường là ăn chữ nhãn cạnh crop.
    if len(primary_tokens) >= 4 and len(norm_tokens) > len(primary_tokens) + 1 and norm_artifacts >= primary_artifacts:
        return True

    return False


def _ocr_tokens(text: str) -> List[str]:
    """Token OCR đã bỏ dấu để so sánh bao hàm, không phụ thuộc tên riêng."""
    return re.findall(r'[a-z0-9]+', _strip_vietnamese_accents(text or "").lower())


def _contains_token_subsequence(container: List[str], needle: List[str]) -> bool:
    if not needle or len(needle) > len(container):
        return False
    for start in range(0, len(container) - len(needle) + 1):
        if container[start:start + len(needle)] == needle:
            return True
    return False


def _normalized_adds_suspicious_text(primary_text: str, norm_text: str) -> bool:
    """
    Trả True khi normalized chỉ là primary cộng thêm text ngoài mép crop.
    Rule tổng quát: primary đã nằm nguyên trong normalized, normalized có phần dư ở đầu/cuối,
    thì coi phần dư là nhiễu trừ khi primary đang rỗng/rác.
    """
    primary_tokens = _ocr_tokens(primary_text)
    norm_tokens = _ocr_tokens(norm_text)
    if not primary_tokens or not norm_tokens:
        return False
    if len(norm_tokens) <= len(primary_tokens):
        return False

    match_start = -1
    for start in range(0, len(norm_tokens) - len(primary_tokens) + 1):
        if norm_tokens[start:start + len(primary_tokens)] == primary_tokens:
            match_start = start
            break

    if match_start < 0:
        return False

    leading_extra = norm_tokens[:match_start]
    trailing_extra = norm_tokens[match_start + len(primary_tokens):]

    # Primary đã là chuỗi con đầy đủ, normalized chỉ thêm text ở mép crop: giữ primary.
    # Bắt case "Lightness / Thư viện số..." và mọi nhiễu tương tự, không hardcode.
    if leading_extra or trailing_extra:
        return True

    extra_tokens = norm_tokens.copy()
    for token in primary_tokens:
        try:
            extra_tokens.remove(token)
        except ValueError:
            pass

    if any(len(tok) <= 1 for tok in extra_tokens):
        return True

    for segment in re.split(r'\s*/\s*', norm_text or ""):
        seg_words = re.findall(r'[A-Za-zÀ-ỹ0-9]+', segment)
        if len(seg_words) > 1 and (len(seg_words[0]) == 1 or len(seg_words[-1]) == 1):
            return True

    return False


def _segment_has_strong_signal(segment: str) -> bool:
    """Segment có khả năng là tên thật: nhiều từ, có dấu Việt, số địa chỉ, hoặc chữ hoa/thương hiệu ngắn."""
    words = re.findall(r'[A-Za-zÀ-ỹ0-9]+', segment or "")
    if not words:
        return False
    if len(words) >= 2:
        return True
    token = words[0]
    if re.search(r'[À-ỹ]', token) or re.search(r'\d', token):
        return True
    if token.isupper() and 2 <= len(token) <= 6:
        return True
    return False


def _is_weak_edge_segment(segment: str) -> bool:
    """Nhận diện segment rìa yếu sinh từ chữ/icon nhãn lân cận, không dựa tên riêng."""
    s = (segment or "").strip()
    words = re.findall(r'[A-Za-zÀ-ỹ0-9]+', s)
    if len(words) != 1:
        return False
    token = words[0]
    clean = _strip_vietnamese_accents(token).lower()
    if len(clean) <= 3:
        return True
    if re.search(r'[À-ỹ\d]', token):
        return False
    # Lowercase 1 từ ở rìa thường là mảnh chữ nhãn khác: reverses, tybrid...
    if token[:1].islower() and len(clean) >= 5:
        return True
    # Titlecase dài kết thúc bằng đuôi OCR artifact như "Priviness".
    # Giữ an toàn vì chỉ áp dụng khi token nằm ở mép và phần còn lại có tín hiệu mạnh.
    if token[:1].isupper() and token[1:].islower() and len(clean) >= 8 and clean.endswith("iness"):
        return True
    # ALLCAPS dài không phải acronym ngắn thường là mảnh OCR cạnh crop.
    if token.isupper() and len(clean) > 6:
        return True
    return False


def _clean_ocr_edge_segments(text: str) -> str:
    """Loại segment rác ở đầu/cuối khi kết quả có nhiều segment và lõi đủ mạnh."""
    parts = [p.strip() for p in re.split(r'\s*/\s*', text or "") if p.strip()]
    if len(parts) < 2:
        return (text or "").strip()

    kept = parts[:]
    while len(kept) >= 2 and _is_weak_edge_segment(kept[0]) and any(_segment_has_strong_signal(p) for p in kept[1:]):
        kept.pop(0)
    while len(kept) >= 2 and _is_weak_edge_segment(kept[-1]) and any(_segment_has_strong_signal(p) for p in kept[:-1]):
        kept.pop()

    return " / ".join(kept)


def _recognize_text_paddle(cv_img: np.ndarray) -> str:
    """Fallback recognition bằng PaddleOCR trên crop đã chọn."""
    detector = _get_paddle_text_detector()
    if detector is None or cv_img is None or cv_img.size == 0:
        return ""
    try:
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        results = detector.predict(rgb)
        texts = []
        for res in results if isinstance(results, list) else [results]:
            if isinstance(res, dict):
                for key in ("rec_texts", "texts"):
                    vals = res.get(key)
                    if isinstance(vals, list):
                        texts.extend(str(v).strip() for v in vals if str(v).strip())
                if "rec_text" in res and str(res["rec_text"]).strip():
                    texts.append(str(res["rec_text"]).strip())
            elif isinstance(res, (list, tuple)):
                for item in res:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        rec = item[1]
                        if isinstance(rec, (list, tuple)) and rec:
                            texts.append(str(rec[0]).strip())
                        elif isinstance(rec, str):
                            texts.append(rec.strip())
        cleaned = []
        for t in texts:
            t = _clean_junk_words(_clean_spelling(t))
            if t and not _is_junk_line(t):
                cleaned.append(t)
        return " / ".join(cleaned)
    except Exception as exc:
        logger.debug("PaddleOCR recognition fallback error: %s", exc)
        return ""


def normalize_ocr_background(crop_img: np.ndarray) -> np.ndarray:
    """Chuẩn hóa nền crop sang màu ngà sáng, giữ chữ màu/đậm để OCR đọc ổn định hơn."""
    if crop_img is None or crop_img.size == 0:
        return crop_img
    try:
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        # Giữ chữ màu đậm (V < 165) hoặc chữ có màu sắc sặc sỡ (S > 80)
        # Loại bỏ các vùng nền xám/xanh nhạt của đường và nền đất ngà
        text_mask = (v < 165) | (s > 80)
        text_mask = cv2.dilate(text_mask.astype(np.uint8), np.ones((2, 2), np.uint8), iterations=1).astype(bool)
        clean_bg = np.array([245, 242, 232], dtype=np.uint8)  # BGR ngà nhạt giống crop đọc tốt

        out = crop_img.copy()
        out[~text_mask] = clean_bg
        return out
    except Exception:
        return crop_img


def split_crop_into_lines_normalized(crop_img: np.ndarray, scale: float = 1.0) -> List[np.ndarray]:
    """Tách crop thành các dòng chữ đơn lẻ dùng ngưỡng chiều cao tối thiểu 3*scale."""
    h_sz, w_sz = crop_img.shape[:2]
    if h_sz < int(12 * scale):
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
                    if end_y - start_y >= int(3 * scale):
                        bands.append((start_y, end_y))
                    in_band = False
                    
        if in_band:
            if h_sz - start_y >= int(3 * scale):
                bands.append((start_y, h_sz))
                
        if len(bands) <= 1:
            return [crop_img]
            
        line_crops = []
        pad_y = max(4, int(4 * scale))
        for sy, ey in bands:
            y1 = max(0, sy - pad_y)
            y2 = min(h_sz, ey + pad_y)
            line_crops.append(crop_img[y1:y2, :])
            
        return line_crops
        
    except Exception as e:
        logger.debug("Lỗi khi chia dòng OCR normalized: %s", e)
        return [crop_img]


def _recognize_text_crop_vietocr_normalized(
    cv_img: np.ndarray,
    bbox: List[float],
    icon_side: str = "left",
    scale: float = 1.0,
    cx: float = None,
    cy: float = None
) -> str:
    """Hàm OCR phụ chạy trên nền được chuẩn hóa hoàn toàn và xử lý icon triệt để."""
    predictor = _get_vietocr_predictor()
    if predictor is None or cv_img is None:
        return ""

    try:
        h_img, w_img = cv_img.shape[:2]
        x1_orig, y1_orig, x2_orig, y2_orig = map(int, bbox)
        
        pad = int(16 * scale)
        y1 = max(0, y1_orig - pad)
        y2 = min(h_img, y2_orig + pad)
        x1 = max(0, x1_orig - pad)
        x2 = min(w_img, x2_orig + pad)
        
        raw_crop = cv_img[y1:y2, x1:x2]
        if raw_crop.size == 0:
            return ""

        # 1. Khử nền bằng logic cải tiến
        base_norm_img = normalize_ocr_background(raw_crop)
        h_rc, w_rc = base_norm_img.shape[:2]

        bg_color = [245, 242, 232]  # BGR ngà nhạt giống crop đọc tốt
        cx_local = (cx - x1) if cx is not None else None
        cy_local = (cy - y1) if cy is not None else None

        def _run_normalized_variant(mask_icon: bool) -> str:
            """Chạy OCR normalized với/không mask icon để tránh cắt mất chữ đầu."""
            norm_img = base_norm_img.copy()

            if mask_icon and cx_local is not None and cy_local is not None:
                if icon_side == "left":
                    mask_w = max(0, min(w_rc, int(cx_local + 6.5 * scale)))
                    norm_img[:, 0:mask_w] = bg_color
                elif icon_side == "right":
                    mask_x = max(0, min(w_rc, int(cx_local - 6.5 * scale)))
                    norm_img[:, mask_x:w_rc] = bg_color
                elif icon_side == "top":
                    mask_h = max(0, min(h_rc, int(cy_local + 6.5 * scale)))
                    norm_img[0:mask_h, :] = bg_color

            tx1, ty1, tx2, ty2 = detect_text_area(norm_img, scale)
            tx1_safe = max(0, tx1 - int(2 * scale))
            tx2_safe = min(w_rc, tx2 + int(4 * scale))
            crop = norm_img[ty1:ty2, tx1_safe:tx2_safe]
            if crop.size == 0:
                return ""

            line_crops = split_crop_into_lines_normalized(crop, scale)
            variant_texts = []
            for line_crop in line_crops:
                if line_crop.size == 0:
                    continue
                line_rgb = cv2.cvtColor(line_crop, cv2.COLOR_BGR2RGB)
                bg_color_line = line_crop[0, 0].tolist()
                pad_h = max(4, int(6 * scale))
                pad_w = max(8, int(12 * scale))
                padded_line = cv2.copyMakeBorder(
                    line_rgb,
                    pad_h, pad_h, pad_w, pad_w,
                    cv2.BORDER_CONSTANT,
                    value=bg_color_line
                )

                def _clean_ocr_line(text: str) -> str:
                    # Strip any rating-like prefix ending with a number in parentheses, e.g. "4.14 (382) - " or "J4x(382) - "
                    text_clean = re.sub(r'^.*?\(\d+(?:[.,]\d+)?\s*[KkM]?[+-]?\)\s*(?:[-·•*]\s*)?', '', (text or '').strip()).strip()
                    text_clean = _clean_junk_words(text_clean)
                    if text_clean and not _is_junk_line(text_clean):
                        return text_clean
                    return ""

                def _line_score(text: str) -> int:
                    return _score_ocr_text_quality(text)

                candidates = []
                for fx in (1, 2, 3):
                    if fx == 1:
                        candidate_img = padded_line
                    else:
                        candidate_img = cv2.resize(
                            padded_line,
                            None,
                            fx=fx,
                            fy=fx,
                            interpolation=cv2.INTER_CUBIC
                        )
                    raw_text = (predictor.predict(Image.fromarray(candidate_img)) or "").strip()
                    clean_text = _clean_ocr_line(raw_text)
                    if clean_text:
                        score = _score_ocr_text_quality(clean_text, fx)
                        candidates.append((score, clean_text))

                if candidates:
                    text_clean = max(candidates, key=lambda x: x[0])[1]
                    variant_texts.append(text_clean)

            return " / ".join(variant_texts)

        masked_text = _run_normalized_variant(mask_icon=True)
        unmasked_text = _run_normalized_variant(mask_icon=False)

        def _variant_score(text: str) -> int:
            clean_len = len(re.sub(r'[^A-Za-zÀ-ỹ0-9]', '', text or ""))
            line_bonus = 12 * max(0, (text or "").count(" / "))
            bad_penalty = 80 if _looks_like_bad_ocr(text) else 0
            return clean_len + line_bonus - bad_penalty

        # Nếu crop đã sạch icon, bản không mask thường giữ được chữ đầu (Little, sách, Mặn...).
        # Chỉ chọn unmasked khi nó tốt hơn rõ ràng, tránh ảnh hưởng các POI có icon thật.
        if unmasked_text and _variant_score(unmasked_text) >= _variant_score(masked_text) + 5:
            return unmasked_text

        return masked_text
    except Exception as exc:
        logger.debug("Lỗi trong recognize_text_crop_vietocr_normalized: %s", exc)
        return ""


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

    # Tiền xử lý làm sạch nền:
    # 1. Tính màu nền chủ đạo bằng median BGR của raw_crop
    bg_color = np.median(raw_crop, axis=(0, 1)).astype(int).tolist()

    # 2. Tạo bản sao sạch và tô đè màu nền lên vùng padding 10px ngoài
    crop_clean = raw_crop.copy()
    h_rc, w_rc = crop_clean.shape[:2]
    border_w = int(10 * scale)
    if border_w > 0:
        if border_w < h_rc:
            crop_clean[0:border_w, :] = bg_color
            crop_clean[h_rc - border_w:, :] = bg_color
        if border_w < w_rc:
            crop_clean[:, 0:border_w] = bg_color
            crop_clean[:, w_rc - border_w:] = bg_color

    def _ocr_from_prepared_crop(prepared_crop: np.ndarray) -> str:
        tx1, ty1, tx2, ty2 = detect_text_area(prepared_crop, scale)
        # Nới nhẹ mép trái để tránh mất nét dọc đầu chữ như H/L/T khi crop sát icon.
        tx1_safe = max(0, tx1 - int(6 * scale))
        tx2_safe = min(prepared_crop.shape[1], tx2 + int(3 * scale))
        crop_local = prepared_crop[ty1:ty2, tx1_safe:tx2_safe]

        if crop_local.size <= 0:
            return ""

        try:
            # Tách crop thành các dòng chữ đơn lẻ
            line_crops = split_crop_into_lines(crop_local, scale)
            
            texts = []
            for line_crop in line_crops:
                if line_crop.size == 0:
                    continue
                    
                # Thêm viền sạch (padding) xung quanh ảnh với màu nền này để tránh lỗi biên của OCR
                bg_color_line = line_crop[0, 0].tolist()
                pad_h = max(4, int(6 * scale))
                pad_w = max(10, int(16 * scale))
                padded_line = cv2.copyMakeBorder(
                    line_crop,
                    pad_h, pad_h, pad_w, pad_w,
                    cv2.BORDER_CONSTANT,
                    value=bg_color_line
                )
                
                rgb_line = cv2.cvtColor(padded_line, cv2.COLOR_BGR2RGB)
                
                # Check 1x, 2x, 3x scales just like the normalized path
                candidates = []
                for fx in (1, 2, 3):
                    im = rgb_line if fx == 1 else cv2.resize(rgb_line, None, fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
                    raw_text = (predictor.predict(Image.fromarray(im)) or "").strip()
                    text_clean = re.sub(r'^.*?\(\d+(?:[.,]\d+)?\s*[KkM]?[+-]?\)\s*(?:[-·•*]\s*)?', '', raw_text).strip()
                    text_clean = _clean_junk_words(text_clean)
                    if text_clean:
                        score = _score_ocr_text_quality(text_clean, fx)
                        candidates.append((score, text_clean))
                        
                if candidates:
                    best_line_text = max(candidates, key=lambda x: x[0])[1]
                    if best_line_text and not _is_junk_line(best_line_text):
                        texts.append(best_line_text)
            return " / ".join(texts)
        except Exception as e:
            logger.debug("Primary VietOCR error: %s", e)
            return ""

    # 3. Vẽ đè vòng tròn màu nền lên vùng icon nếu có tọa độ cx, cy.
    # Chạy thêm bản không mask để tránh ăn mất chữ đầu khi bbox/icon sát chữ.
    crop_masked = crop_clean.copy()
    if cx is not None and cy is not None:
        cx_local = cx - x1
        cy_local = cy - y1
        r = int(16 * scale)
        cv2.circle(crop_masked, (int(cx_local), int(cy_local)), r, bg_color, -1)

    primary_masked = _ocr_from_prepared_crop(crop_masked)
    primary_unmasked = _ocr_from_prepared_crop(crop_clean)

    def _should_use_unmasked(masked_text: str, unmasked_text: str) -> bool:
        if not masked_text or not unmasked_text:
            return bool(unmasked_text and not masked_text)
        if _looks_like_bad_ocr(unmasked_text):
            return False

        masked_tokens = _ocr_tokens(masked_text)
        unmasked_tokens = _ocr_tokens(unmasked_text)
        if not masked_tokens or not unmasked_tokens:
            return False

        # Nếu cùng số token, chỉ cho unmasked thắng khi nó thật sự khôi phục token bị cắt đầu.
        # Ví dụ cần cứu: tybrid -> hybrid (unmasked dài hơn 1 ký tự và chứa masked làm suffix).
        # Ví dụ phải giữ: tam -> vi tam (unmasked chèn token nhiễu), sai gon -> sai gon giữ nguyên.
        if len(masked_tokens) == len(unmasked_tokens):
            masked_raw_tokens = re.findall(r'[A-Za-zÀ-ỹ0-9]+', masked_text)
            improved_prefix = False
            worsened = False
            for idx, (m_tok, u_tok) in enumerate(zip(masked_tokens, unmasked_tokens)):
                if m_tok == u_tok:
                    continue
                raw_m = masked_raw_tokens[idx] if idx < len(masked_raw_tokens) else ""
                # Chỉ cứu token bắt đầu lowercase: tybrid -> hybrid.
                # Không sửa token viết hoa hợp lệ: Efora -> LEfora.
                if raw_m[:1].islower() and len(u_tok) == len(m_tok) + 1 and u_tok.endswith(m_tok):
                    improved_prefix = True
                    continue
                worsened = True
                break
            return improved_prefix and not worsened

        # Nếu unmasked thêm token, coi là nhiễu icon/chữ lân cận trừ khi masked gần như rỗng/rác.
        if len(unmasked_tokens) > len(masked_tokens):
            return _looks_like_bad_ocr(masked_text)

        return False

    if _should_use_unmasked(primary_masked, primary_unmasked):
        primary_text = primary_unmasked
        logger.info("  [OCR primary unmasked chosen] Masked='%s' -> Unmasked='%s'", primary_masked, primary_unmasked)
    else:
        primary_text = primary_masked

    # 4. Chạy luồng normalized OCR làm fallback
    norm_text = _recognize_text_crop_vietocr_normalized(cv_img, bbox, icon_side, scale, cx, cy)
    
    # 5. So sánh chất lượng và chọn kết quả tốt nhất bằng score tổng quát, không keyword.
    def _candidate_score(text: str, source: str) -> float:
        cleaned = _clean_final_ocr_text(text)
        if not cleaned:
            return -9999
        score = _score_ocr_text_quality(cleaned)
        # Không cộng bonus riêng cho normalized: log thực tế cho thấy normalized hay dài/sạch hơn nhưng đổi sai chữ gốc
        # như `Bưu` -> `Bir`, `Cổng` -> `Cống`, `19th-century` -> `9th-censury`.
        return score

    candidates = []
    for source, raw_text in (("primary", primary_text), ("normalized", norm_text)):
        cleaned = _clean_final_ocr_text(raw_text)
        if cleaned:
            candidates.append((_candidate_score(raw_text, source), source, cleaned, raw_text))

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_source, selected_text, selected_raw = candidates[0]
        primary_candidate = next((item for item in candidates if item[1] == "primary"), None)
        if (
            best_source == "normalized"
            and primary_candidate
            and not _looks_like_bad_ocr(primary_text)
            and _normalized_adds_suspicious_text(primary_text, norm_text)
        ):
            _, _, primary_cleaned, _ = primary_candidate
            selected_text = primary_cleaned
            best_source = "primary"
            logger.info(
                "  [OCR keep primary] Primary='%s' rejected suspicious-extra Normalized='%s'",
                primary_cleaned,
                norm_text,
            )
        if best_source == "normalized" and primary_candidate and not _looks_like_bad_ocr(primary_text):
            primary_score, _, primary_cleaned, _ = primary_candidate
            norm_gain = best_score - primary_score
            if norm_gain < 9:
                selected_text = primary_cleaned
                best_source = "primary"
                logger.info(
                    "  [OCR keep primary] Primary='%s' rejected non-clear Normalized='%s' (gain=%.1f)",
                    primary_cleaned,
                    selected_raw,
                    norm_gain,
                )
        is_normalized_chosen = best_source == "normalized"
        if best_source == "primary" and norm_text and _normalized_regresses_quality(primary_text, norm_text):
            logger.info("  [OCR keep primary] Primary='%s' rejected lower-quality Normalized='%s'", primary_text, norm_text)
        elif best_source == "normalized":
            logger.info("  [OCR normalized chosen] Primary='%s' -> Normalized='%s'", primary_text, selected_text)
    elif norm_text:
        selected_text = _clean_final_ocr_text(norm_text)
        is_normalized_chosen = True
    else:
        selected_text = _clean_final_ocr_text(primary_text)
        is_normalized_chosen = False

    if selected_text and primary_text and selected_text != primary_text:
        merged_text = _merge_best_diacritics(primary_text, selected_text)
        merged_text = _clean_final_ocr_text(merged_text)
        if merged_text and merged_text != selected_text:
            logger.info("  [OCR merge diacritics] Selected='%s' + Primary='%s' -> '%s'", selected_text, primary_text, merged_text)
            selected_text = merged_text
        
    # 6. Fallback sang PaddleOCR nếu cả hai luồng đều lỗi/rác
    if not selected_text or _looks_like_bad_ocr(selected_text) or _junk_token_count(selected_text) > 0:
        tx1, ty1, tx2, ty2 = detect_text_area(crop_masked, scale)
        crop_for_paddle = crop_masked[ty1:ty2, max(0, tx1 - int(6 * scale)):tx2]
        paddle_text = _clean_final_ocr_text(_recognize_text_paddle(crop_for_paddle)) if crop_for_paddle.size > 0 else ""
        if paddle_text and not _looks_like_bad_ocr(paddle_text) and _junk_token_count(paddle_text) == 0:
            logger.info("  [OCR fallback] VietOCR selected='%s' -> PaddleOCR='%s'", selected_text, paddle_text)
            return paddle_text

    return _clean_final_ocr_text(selected_text)


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
        name_cleaned = _clean_ocr_edge_segments(name)
        if name_cleaned != name:
            logger.info("  [OCR edge cleanup] '%s' -> '%s'", name, name_cleaned)
            name = name_cleaned
        
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

def save_poi_crop(image_bytes: bytes, poi: dict, output_path: str, scale: float = 1.0):
    """Lưu ảnh crop POI đẹp: chỉ text, padding đều, không lộ icon, không cắt chữ."""
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

        # Lấy vùng rộng hơn bbox để detect đủ chữ sát biên trước khi crop lại đẹp.
        outer_pad = max(10, int(12 * scale))
        y1 = max(0, t - outer_pad)
        y2 = min(h_img, t + h + outer_pad)
        x1 = max(0, l - outer_pad)
        x2 = min(w_img, l + w + outer_pad)

        raw_crop = img[y1:y2, x1:x2]
        if raw_crop.size == 0:
            return

        bg_color = np.median(raw_crop, axis=(0, 1)).astype(int).tolist()
        h_rc, w_rc = raw_crop.shape[:2]

        cx = poi.get("x")
        cy = poi.get("y")
        icon_side = poi.get("icon_side", "left")
        cx_local = (cx - x1) if cx is not None else None
        cy_local = (cy - y1) if cy is not None else None

        def _clean_outer_border(img_part: np.ndarray):
            """Dọn viền của ảnh detect để không bắt nhiễu map ở vùng lấy rộng."""
            border_w = max(8, int(10 * scale))
            hp, wp = img_part.shape[:2]
            if border_w > 0:
                if border_w < hp:
                    img_part[0:border_w, :] = bg_color
                    img_part[hp - border_w:, :] = bg_color
                if border_w < wp:
                    img_part[:, 0:border_w] = bg_color
                    img_part[:, wp - border_w:] = bg_color

        def _mask_icon_for_detection(img_part: np.ndarray):
            """Mask icon chỉ để tìm text box, không dùng ảnh này làm crop cuối."""
            if img_part is None or img_part.size == 0 or cx_local is None or cy_local is None:
                return
            hp, wp = img_part.shape[:2]
            if icon_side == "left":
                mask_w = max(0, min(wp, int(cx_local + 7.5 * scale)))
                img_part[:, :mask_w] = bg_color
            elif icon_side == "right":
                mask_x = max(0, min(wp, int(cx_local - 7.5 * scale)))
                img_part[:, mask_x:] = bg_color
            elif icon_side == "top":
                mask_h = max(0, min(hp, int(cy_local + 7.5 * scale)))
                img_part[:mask_h, :] = bg_color

        # Ảnh detect có thể bị mask icon; crop cuối lấy từ raw_crop để không mất nét chữ.
        detect_img = raw_crop.copy()
        _clean_outer_border(detect_img)
        _mask_icon_for_detection(detect_img)

        tx1, ty1, tx2, ty2 = detect_text_area(detect_img, scale)

        # Padding đẹp 4 phía. Giữ phải rộng hơn vì chữ cuối thường sát biên/nhỏ.
        pad_left = max(6, int(8 * scale))
        pad_right = max(10, int(14 * scale))
        pad_top = max(4, int(5 * scale))
        pad_bottom = max(5, int(6 * scale))
        raw_tx1 = tx1
        raw_ty1 = ty1
        raw_tx2 = tx2
        raw_ty2 = ty2
        tx1 = max(0, raw_tx1 - pad_left)
        ty1 = max(0, raw_ty1 - pad_top)
        tx2 = min(w_rc, raw_tx2 + pad_right)
        ty2 = min(h_rc, raw_ty2 + pad_bottom)

        crop = raw_crop[ty1:ty2, tx1:tx2].copy()
        if crop.size == 0:
            return

        # Dọn icon sót trong vùng padding, không tô vào vùng text đã detect.
        final_h, final_w = crop.shape[:2]
        text_left_in_crop = max(0, raw_tx1 - tx1)
        text_right_in_crop = min(final_w, raw_tx2 - tx1)
        text_top_in_crop = max(0, raw_ty1 - ty1)

        if cx_local is not None and cy_local is not None:
            # Dọn vùng icon theo tọa độ thật, gồm cả icon trắng/bóng mờ low-saturation.
            # HSV-only không bắt được các mảng trắng/xám nên vẫn còn dấu vết.
            if icon_side == "left":
                icon_edge_in_crop = int(cx_local + 12.5 * scale) - tx1
                clean_w = max(0, min(final_w, icon_edge_in_crop))
                if clean_w > 0:
                    crop[:, :clean_w] = bg_color
            elif icon_side == "right":
                icon_edge_in_crop = int(cx_local - 12.5 * scale) - tx1
                clean_x = max(0, min(final_w, icon_edge_in_crop))
                if clean_x < final_w:
                    crop[:, clean_x:] = bg_color
            elif icon_side == "top":
                icon_edge_in_crop = int(cy_local + 12.5 * scale) - ty1
                clean_h = max(0, min(final_h, icon_edge_in_crop))
                if clean_h > 0:
                    crop[:clean_h, :] = bg_color

            # Dọn thêm mảng màu icon nếu còn chạm mép ngoài vùng hình chữ nhật.
            def _erase_edge_connected_icon(side: str):
                hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                s = hsv_crop[:, :, 1]
                v = hsv_crop[:, :, 2]
                icon_mask = ((s > 35) & (v > 60)).astype(np.uint8) * 255
                kernel = np.ones((3, 3), np.uint8)
                icon_mask = cv2.dilate(icon_mask, kernel, iterations=1)
                scan_w = min(final_w, max(18, int(28 * scale)))
                scan_h = min(final_h, max(18, int(28 * scale)))
                edge_tol = max(2, int(3 * scale))
                clean_mask = np.zeros((final_h, final_w), dtype=np.uint8)

                if side == "left":
                    roi = icon_mask[:, :scan_w]
                    x_base, y_base = 0, 0
                elif side == "right":
                    roi = icon_mask[:, final_w - scan_w:]
                    x_base, y_base = final_w - scan_w, 0
                elif side == "top":
                    roi = icon_mask[:scan_h, :]
                    x_base, y_base = 0, 0
                else:
                    return

                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(roi, 8)
                for label in range(1, num_labels):
                    x, y, w_box, h_box, area = stats[label]
                    if area < max(2, int(3 * scale * scale)):
                        continue
                    touches_edge = (
                        (side == "left" and x <= edge_tol) or
                        (side == "right" and x + w_box >= roi.shape[1] - edge_tol) or
                        (side == "top" and y <= edge_tol)
                    )
                    if touches_edge:
                        component = (labels == label).astype(np.uint8) * 255
                        clean_mask[y_base:y_base + roi.shape[0], x_base:x_base + roi.shape[1]] |= component

                if np.any(clean_mask):
                    clean_mask[:] = cv2.dilate(clean_mask, kernel, iterations=1)
                    crop[clean_mask > 0] = bg_color

            if icon_side == "left":
                _erase_edge_connected_icon("left")
            elif icon_side == "right":
                _erase_edge_connected_icon("right")
            elif icon_side == "top":
                _erase_edge_connected_icon("top")

        # Làm sạch nền viền rất mỏng trên/dưới; không tô trái/phải để tránh mất nét đầu/cuối.
        edge_pad = max(1, int(1 * scale))
        if crop.shape[0] > 2 * edge_pad:
            crop[0:edge_pad, :] = bg_color
            crop[crop.shape[0] - edge_pad:, :] = bg_color

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, crop)
        logger.info("  [Crop saved] %s", os.path.basename(output_path))
    except Exception as e:
        logger.warning("Không thể lưu ảnh crop POI: %s", e)
