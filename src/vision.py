# =============================================================
# vision.py — Nhận diện POI bằng Tesseract OCR & OpenCV
#
# Nhận screenshot bytes → lưu ảnh tạm → Tiền xử lý 2x & OCR
# → Nhận diện và định vị tâm icon/chữ → Trả về danh sách POI.
# Không sử dụng LLM.
# =============================================================

import asyncio
import csv
import io
import logging
import os
import re
import subprocess
import tempfile
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

try:
    from config import (
        OCR_ENGINE,
        VIETOCR_MODEL,
        VIETOCR_DEVICE,
        OCR_TEXT_PAD_PX,
        OCR_HORIZONTAL_GAP_MAX,
        OCR_ICON_Y_ALIGN_RATIO,
    )
except Exception:
    OCR_ENGINE = "tesseract"
    VIETOCR_MODEL = "vgg_transformer"
    VIETOCR_DEVICE = "cpu"
    OCR_TEXT_PAD_PX = 4
    OCR_HORIZONTAL_GAP_MAX = 10
    OCR_ICON_Y_ALIGN_RATIO = 0.7

_VIETOCR_PREDICTOR = None
_VIETOCR_LOAD_FAILED = False

_last_candidate_icons = []
_last_valid_lines = []
_last_line_to_icon = {}
_last_potential_line_matches = []


# ── Bộ lọc tên loại hình POI chung chung (trích từ bản cũ) ──────────────
_GENERIC_POI_NAMES = {
    "cafe", "coffee", "shop", "store", "restaurant", "hotel", "bank", "market",
    "church", "school", "park", "pharmacy", "clinic", "spa", "gym", "bar",
    "pub", "hostel", "supermarket", "mall", "center", "centre", "tower",
    "building", "office", "station", "post office", "post_office", "landmark",
    "nhà hàng", "quán ăn", "cà phê", "ngân hàng", "khách sạn", "trường học",
    "bệnh viện", "chợ", "công viên", "nhà thờ", "siêu thị", "tòa nhà", "văn phòng",
    "bưu điện", "trụ sở", "cửa hàng", "cửa hiệu", "hiệu thuốc", "quầy thuốc",
    # Mở rộng các từ loại hình tiếng Việt/Anh chung chung khác
    "rau sạch", "trái cây", "tạp hóa", "quán nước", "trà sữa", "ăn vặt", "bánh mì",
    "cửa hàng tiện lợi", "siêu thị mini", "atm", "rạp chiếu phim", "nhà sách",
    "hiệu sách", "lịch", "lịch sử", "địa điểm", "bản đồ", "tòa đại sứ",
    "lãnh sự", "lãnh sự quán", "đại sứ quán", "tổng lãnh sự quán", "ủy ban",
    "ủy ban nhân dân", "ubnd", "trụ sở ubnd", "công an", "đồn công an",
    "trạm y tế", "nhà khách", "nhà nghỉ", "biệt thự", "chung cư",
    # Thêm các từ rác/chung chung khi đứng độc lập (thường do tách dòng)
    "soon", "coming soon", "open soon", "tương", "tượng", "hoa binh", "hòa bình"
}

# ── Danh sách tên quốc gia để loại bỏ nhãn quốc gia độc lập (do rã dòng từ đại sứ quán/lãnh sự quán) ──
_COUNTRY_NAMES = {
    "hoa kỳ", "pháp", "anh", "đức", "nhật bản", "hàn quốc", "việt nam", "trung quốc", 
    "nga", "singapore", "thái lan", "malaysia", "campuchia", "lào", "úc", "italy", "ý", 
    "tây ban nha", "bồ đào nha", "thuỵ sĩ", "thuỵ điển", "bỉ", "hà lan", "ấn độ", "canada"
}

def _is_generic_name(name: str) -> bool:
    """Trả về True nếu nhãn chỉ chứa một từ thể loại chung chung không có tên riêng."""
    name_clean = name.strip()
    if not name_clean:
        return True
    name_lower = name_clean.lower()
    if name_lower in _GENERIC_POI_NAMES:
        return True
    return False


def _get_region_color(img: np.ndarray, bbox: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
    """
    Trích xuất màu BGR đại diện của vùng ảnh bbox [x1, y1, x2, y2].
    Chỉ xét các điểm ảnh không phải là nền sáng (ví dụ: độ sáng grayscale < 210)
    để lọc ra nét chữ hoặc nét của biểu tượng.
    """
    x1, y1, x2, y2 = map(int, bbox)
    h_img, w_img = img.shape[:2]
    
    # Kẹp tọa độ trong phạm vi kích thước ảnh
    x1 = max(0, min(x1, w_img - 1))
    x2 = max(0, min(x2, w_img - 1))
    y1 = max(0, min(y1, h_img - 1))
    y2 = max(0, min(y2, h_img - 1))
    
    if x2 <= x1 or y2 <= y1:
        return None
        
    crop = img[y1:y2, x1:x2]
    # Chuyển sang ảnh xám để tìm điểm ảnh tối đại diện cho nét chữ/icon
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # Lọc bỏ nền sáng (OpenStreetMap nền thường rất sáng > 210)
    mask = gray_crop < 210
    
    if not np.any(mask):
        # Nếu toàn bộ vùng là nền sáng, lấy trung bình toàn bộ vùng
        avg_bgr = np.mean(crop, axis=(0, 1))
    else:
        avg_bgr = np.mean(crop[mask], axis=0)
        
    return avg_bgr


def _colors_are_similar(color1: Optional[np.ndarray], color2: Optional[np.ndarray], thresh_h: int = 20, thresh_s: int = 60, thresh_v: int = 60) -> bool:
    """
    So sánh độ tương đồng của hai màu trong không gian màu HSV.
    """
    if color1 is None or color2 is None:
        return False
        
    # Chuyển đổi màu từ BGR sang HSV (OpenCV nhận uint8 định dạng 1x1 pixel)
    c1_hsv = cv2.cvtColor(np.uint8([[color1]]), cv2.COLOR_BGR2HSV)[0][0]
    c2_hsv = cv2.cvtColor(np.uint8([[color2]]), cv2.COLOR_BGR2HSV)[0][0]
    
    h1, s1, v1 = c1_hsv
    h2, s2, v2 = c2_hsv
    
    # Kiểm tra xem màu có phải là tông xám/đen/tối (không rõ sắc độ) hay không
    is_dark_or_gray1 = s1 < 40 or v1 < 50
    is_dark_or_gray2 = s2 < 40 or v2 < 50
    
    # Nếu cả hai đều là tông màu xám/đen/tối, coi như tương đồng màu
    if is_dark_or_gray1 and is_dark_or_gray2:
        return True
    # Nếu một bên là tông xám/đen còn bên kia là màu sắc rõ rệt
    if is_dark_or_gray1 != is_dark_or_gray2:
        return False
        
    # So sánh trị số Hue (sắc độ) có tính chất tuần hoàn [0, 180] trong OpenCV
    h_diff = abs(int(h1) - int(h2))
    h_diff = min(h_diff, 180 - h_diff)
    
    # So sánh Saturation (độ bão hòa) và Value (độ sáng)
    s_diff = abs(int(s1) - int(s2))
    v_diff = abs(int(v1) - int(v2))
    
    return h_diff <= thresh_h and s_diff <= thresh_s and v_diff <= thresh_v


# ── Bộ lọc tên đường giao thông (Sửa lỗi khớp không dấu) ─────────────────

def _strip_vietnamese_accents(s: str) -> str:
    """Loại bỏ hoàn toàn dấu tiếng Việt và đưa về chữ thường."""
    s = s.lower()
    s = re.sub(r"[àáảãạăằắẳẵặâầấẩẫậ]", "a", s)
    s = re.sub(r"[èéẻẽẹêềếểễệ]", "e", s)
    s = re.sub(r"[ìíỉĩị]", "i", s)
    s = re.sub(r"[òóỏõọôồốổỗộơờớởỡợ]", "o", s)
    s = re.sub(r"[ùúủũụưừứửữự]", "u", s)
    s = re.sub(r"[ỳýỷỹỵ]", "y", s)
    s = re.sub(r"[đ]", "d", s)
    return s


def _is_street_name(name: str) -> bool:
    """
    Nhận dạng tên đường bằng các biểu thức chính quy và quy tắc tiếng Việt.
    Đã sửa lỗi so khớp không dấu ở phần kiểm tra họ và tiền tố nước ngoài.
    """
    name_clean = name.strip()
    if not name_clean:
        return False
    name_lower = name_clean.lower()

    # ① Từ khóa giao thông rõ ràng (được định nghĩa cụ thể)
    _STREET_KEYWORDS = re.compile(
        r"\b(đường|phố|đại lộ|quốc lộ|tỉnh lộ|liên tỉnh|liên huyện"
        r"|boulevard|avenue|street|road|lane|alley|st\b|rd\b|ave\b|ln\b|blvd\b"
        r"|hẻm|ngõ|kiệt|xóm|thôn\s+\d"
        r"|vòng xoay|nút giao|ngã tư|ngã năm|ngã sáu|ngã bảy|ngã ba"
        r"|cầu\s+\w|cống\s+\w"
        r")\b",
        re.IGNORECASE,
    )

    # ② Cụm địa giới hành chính cần loại bỏ trước khi check (tránh loại nhầm Bưu điện TP, v.v.)
    _FALSE_POSITIVE_PHRASES = re.compile(
        r"\b(thành\s*phố|thành\s*thị|thị\s*xã|thị\s*trấn|tỉnh\s+\w+|quận\s+\d+|huyện\s+\w+)\b",
        re.IGNORECASE,
    )

    # ③ Tên đường có số hiệu (QL1, TL13...)
    _NUMBERED_ROAD = re.compile(
        r"^(ql|tl|đt|nh|d|n|b|c|r|f|g|h|k|p|q|s|t|u|v|w|x|y|z)\s*\d+[a-z]?$",
        re.IGNORECASE,
    )

    check_str = _FALSE_POSITIVE_PHRASES.sub("", name_lower).strip()
    if _STREET_KEYWORDS.search(check_str):
        return True

    no_accent = _strip_vietnamese_accents(check_str)
    if _NUMBERED_ROAD.match(no_accent.strip()):
        return True

    # ④ Heuristic nhận dạng tên đường mang tên người Việt Nam/nước ngoài:
    # 2 đến 5 từ đơn, không chứa từ chỉ loại hình kinh doanh/dịch vụ.
    no_accent_words = no_accent.split()
    if 2 <= len(no_accent_words) <= 5 and re.match(r"^[a-z\s]+$", no_accent):
        _BUSINESS_WORDS = {
            "plaza", "tower", "center", "centre", "mall", "market",
            "coffee", "cafe", "hotel", "hostel", "restaurant", "clinic",
            "hospital", "pharmacy", "bank", "school", "university",
            "college", "church", "temple", "pagoda", "park", "garden",
            "station", "port", "airport", "embassy", "consulate",
            # Tiếng Việt
            "tháp", "trung tâm", "siêu thị", "chợ", "bệnh viện",
            "trường", "đại học", "nhà thờ", "chùa", "công viên",
            "sân bay", "bến xe", "ga", "cảng", "đại sứ quán",
            "khách sạn", "nhà hàng", "quán", "tiệm", "cửa hàng",
        }
        words_set = set(no_accent_words)
        if not words_set.intersection(_BUSINESS_WORDS):
            # Họ người Việt phổ biến dùng đặt tên đường
            _VN_SURNAME_PREFIXES = {
                "nguyen", "tran", "le", "pham", "huynh", "vo", "vu",
                "dang", "bui", "do", "ho", "ngo", "duong", "ly",
                "dinh", "truong", "phan", "luong", "chau", "luu",
                "mai", "to", "cao", "lam", "thai", "trinh", "nhan",
            }
            if no_accent_words[0] in _VN_SURNAME_PREFIXES and len(no_accent_words) >= 2:
                return True
            
            # Tên người nước ngoài đặt tên đường
            _FOREIGN_PREFIXES = {
                "pasteur", "yersin", "calmette", "alexandre",
                "lyautey", "gallieni", "luro",
            }
            if no_accent_words[0] in _FOREIGN_PREFIXES:
                return True

    return False


def _clean_spelling(text: str) -> str:
    """Khắc phục các lỗi dấu và chính tả tiếng Việt phổ biến từ Tesseract OCR."""
    replacements = {
        # Lỗi hai dấu huyền ở nguyên âm đôi (VD: trừờng -> trường)
        r"([tT])rừ[ờơ]ng": r"\1rường",
        r"([đĐ])u\s+ấn\b": r"\1uẩn",
        r"([đĐ])ừ[ờơ]ng": r"\1ường",
        r"([pP])hừ[ờơ]ng": r"\1ường",
        
        # Cả phê / Cà phe / Cả Phê -> Cà phê
        r"\b[cC][ảàa]\s+[pP]h[êe]\b": "Cà phê",
        r"\b[cC]à\s+[pP]h[ếệ]\b": "Cà phê",
        
        # Bánh mì
        r"\b[bB][áa]nh\s+[mM][ìi]\b": "Bánh mì",
        
        # Sài Gòn
        r"\b[sS]ài\s+[gG]òn\b": "Sài Gòn",
        r"\b[sS]ai\s+[gG]on\b": "Sài Gòn",
        r"\bBưu\s+điện\s+trung\s+tâm\s+Sài\s+Gòn\b": "Bưu điện Trung tâm Sài Gòn",
        r"\bBưu\s+điện\s+trung\s+tâm\s+sài\s+Gòn\b": "Bưu điện Trung tâm Sài Gòn",
        r"\bBưu\s+điện\s+trung\s+tâm\s+sài\s+Grand\b": "Bưu điện Trung tâm Sài Gòn",
        r"\bBưu\s+điện\s+trung\s+tâm\s+sài\s+gòn\s+Grand\b": "Bưu điện Trung tâm Sài Gòn",
        
        # ÁO DÀI & ÁO BÀ BA RENTALS
        r'\b[áÁ][oO]\s+[dD][àÀ][iI]\"?\s+[aA][nN][dD]\s+(?:[áÁ][oO]\s+)?[bB][àÀ]\s+[bB][aA]\'?\s+[rR][eE][nN][tT][aA][lL][sS]\b': '"ÁO DÀI" AND "ÁO BÀ BA" RENTALS',
        
        # Lãnh sự quán / Đại sứ quán
        r"\b[tT][ôo]ng\s+[lL]ãnh\b": "Tổng Lãnh",
        r"\b[tT]ổng\s+[lL]ánh\b": "Tổng Lãnh",
        r"\b[lL]ãnh\s+[sS]ứ\s+[qQ]uán\b": "Lãnh sự quán",
        r"\b[tT]ổng\s+[lL]ãnh\s+[sS]ứ\s+[qQ]uán\b": "Tổng Lãnh sự quán",
        r"\b[đĐ]ại\s+[sS]ự\s+[qQ]uán\b": "Đại sứ quán",
        
        # Hoa Kỳ
        r"\b[hH]oa\s+[kK]y\b": "Hoa Kỳ",
        r"\b[hH]oa\s+[kK]ỷ\b": "Hoa Kỳ",
        
        # Đức Bà / Nhà thờ
        r"\b[đĐ]ức\s+[bB][aá]\b": "Đức Bà",
        r"\b[nN]h?[aà]\s+[tT]h[ờo]\b": "Nhà thờ",
        
        # Bưu điện
        r"\b[bB]ưu\s+[đĐ]i[ệẹê]n\b": "Bưu điện",
        r"\b[bB]uu\s+[đĐ]iện\b": "Bưu điện",
        r"\b[bB]ửu\s+[đĐ]iện\b": "Bưu điện",
        
        # Khách sạn
        r"\b[kK]h[áa]ch\s+[sS][ạa]n\b": "Khách sạn",
        
        # Bệnh viện
        r"\b[bB]ệnh\s+[vV]i[ệẹê]n\b": "Bệnh viện",
        
        # Nhà hàng
        r"\b[nN]hà\s+[hH][àa]ng\b": "Nhà hàng",
        
        # Ngân hàng
        r"\b[nN]gân\s+[hH][àa]ng\b": "Ngân hàng",
        
        # Siêu thị
        r"\b[sS]i[êe]u\s+[tT]hị\b": "Siêu thị",
        
        # Trường học / Trường tiểu học
        r"\b[tT]rường\s+[hH]ọc\b": "Trường học",
        r"\b[tT]rường\s+[tT]iêu\b": "trường tiểu",
        
        # Công ty
        r"\b[cC][ôo]ng\s+[tT]y\b": "Công ty",
        
        # Xếp hàng
        r"\b[xX]ếp\s+[hH]ang\b": "xếp hàng",
        
        # Circle K
        r"\b[cC]ircle\s+[kK]\b": "Circle K",
        
        # Lê Duẩn
        r"\bLê\s+đ[uủ]ấn\b": "Lê Duẩn",
        r"\bsự\s+đ[uủ]ấn\b": "Lê Duẩn",
        
        # Sửa lỗi chính tả Tượng/Tương và Hòa Bình
        r"\bTương\s+Đức\s+Bà\b": "Tượng Đức Bà",
        r"\bTương\s+đài\b": "Tượng đài",
        r"\b[hH]oa\s+[bB]inh\b": "Hòa Bình",
        r"\b[hH]oa\s+[bB]ình\b": "Hòa Bình",
        r"\b[hH]òa\s+[bB]inh\b": "Hòa Bình",
        
        # Sửa lỗi chính tả nâng cao
        r"\b[cC]tra\s+[hH]ang\b": "Cửa hàng",
        r"\b[tT]h[öo]f\s+[tT]rang\b": "Thời trang",
        r"\b[nN]h[aà]\s+[tT]h[ờo]\s+[bB]a\b": "Nhà thờ Đức Bà",
        r"\b[nN]h[aà]\s+[tT]h[ờo]\s+[đĐ]ức\s+[bB]a\b": "Nhà thờ Đức Bà",
        r"\b[nN]h?[aà]\s+th?[aà]\s+B[aà]\s+gai\s+Gòn\b": "Nhà thờ Đức Bà Sài Gòn",
        r"\bNhi\s+gong\b": "Nhi đồng",
        r"\bNhi\s+đ[ôo]ng\b": "Nhi đồng",
        r"\b[rR]u\s*[nN]am\s*[dD]or\b": "RuNam D'Or",
        r"\b[rR]u\s*[nN]am\s*[dD]'\s*or\b": "RuNam D'Or",
        r"\bvinaphon[ée]\b": "Vinaphone",
        r"\bTương\s+Đức\s+Binh\b": "Tượng Đức Bà",
        r"\b[sS]ự\s+[qQ]uản\b": "sự quán",
        r"\b[lL]ãnh\s+[sS]ự\s+[qQ]uản\b": "Lãnh sự quán",
        r"\b[đĐ]ại\s+[sS]ự\s+[qQ]uản\b": "Đại sứ quán",
        r"\bGia\s+[đĐ]inh\b": "Gia Định",
    }
    
    cleaned = text
    for pattern, replacement in replacements.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    
    return re.sub(r'\s+', ' ', cleaned).strip()


def _get_vietocr_predictor():
    """Lazy-load VietOCR Predictor một lần; trả None nếu dependency/model lỗi."""
    global _VIETOCR_PREDICTOR, _VIETOCR_LOAD_FAILED
    if _VIETOCR_PREDICTOR is not None:
        return _VIETOCR_PREDICTOR
    if _VIETOCR_LOAD_FAILED or OCR_ENGINE.lower() != "vietocr":
        return None
    try:
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor

        config = Cfg.load_config_from_name(VIETOCR_MODEL)
        config["device"] = VIETOCR_DEVICE
        config["predictor"]["beamsearch"] = True
        _VIETOCR_PREDICTOR = Predictor(config)
        logger.info("[VietOCR] Loaded model=%s device=%s", VIETOCR_MODEL, VIETOCR_DEVICE)
        return _VIETOCR_PREDICTOR
    except Exception as exc:
        _VIETOCR_LOAD_FAILED = True
        logger.warning("[VietOCR] Không load được VietOCR, fallback Tesseract: %s", exc)
        return None


def _recognize_text_crop_vietocr(cv_img: np.ndarray, bbox: Tuple[float, float, float, float], fallback: str = "") -> str:
    """OCR lại crop chữ bằng VietOCR; fallback về text detector nếu cần."""
    predictor = _get_vietocr_predictor()
    if predictor is None or cv_img is None:
        return fallback.strip()

    h_img, w_img = cv_img.shape[:2]
    # Nới rộng padding crop: pad_x = 10 (tránh mất ký tự đầu/cuối), pad_y = 6 (tránh mất dấu tiếng Việt)
    pad_x = 10
    pad_y = 6
    x1, y1, x2, y2 = map(int, bbox)
    x1_pad = max(0, x1 - pad_x)
    y1_pad = max(0, y1 - pad_y)
    x2_pad = min(w_img, x2 + pad_x)
    y2_pad = min(h_img, y2 + pad_y)
    if x2_pad <= x1_pad or y2_pad <= y1_pad:
        return fallback.strip()

    crop = cv_img[y1_pad:y2_pad, x1_pad:x2_pad].copy()
    if crop.size == 0:
        return fallback.strip()

    try:
        # Khử nền bằng cách chuyển sang grayscale và thresholding ở 200
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, crop_thresh = cv2.threshold(crop_gray, 200, 255, cv2.THRESH_BINARY)
        
        # Xóa tất cả các pixel nằm ngoài bounding box thực tế của từ (mặt nạ trắng)
        # để tránh các từ bên cạnh hoặc icon đè vào vùng padding mở rộng gây nhiễu cho VietOCR
        lx1 = x1 - x1_pad
        ly1 = y1 - y1_pad
        lx2 = x2 - x1_pad
        ly2 = y2 - y1_pad
        
        mask = np.zeros(crop_thresh.shape, dtype=np.uint8)
        mask[ly1:ly2, lx1:lx2] = 255
        crop_thresh[mask == 0] = 255

        # VietOCR đọc tốt hơn khi crop chữ nhỏ được phóng nhẹ.
        ch, cw = crop_thresh.shape[:2]
        scale = 2 if max(ch, cw) < 220 else 1
        if scale > 1:
            crop_thresh = cv2.resize(crop_thresh, (cw * scale, ch * scale), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(crop_thresh, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(rgb)
        text = predictor.predict(pil_img)
        text = re.sub(r'\s+', ' ', (text or '')).strip()
        if len(text) >= 2:
            logger.debug("[VietOCR] '%s' -> '%s'", fallback, text)
            return text
    except Exception as exc:
        logger.debug("[VietOCR] Crop OCR lỗi: %s", exc)
    return fallback.strip()


def is_solid_icon(cv_img: np.ndarray, bbox: Tuple[float, float, float, float]) -> bool:
    """
    Kiểm tra xem vùng bbox có phải là một biểu tượng (icon) đặc/đầy hay không.
    Giúp lọc bỏ các đường viền nét chữ hoặc dấu ngoặc kép được OpenCV nhận diện nhầm là icon.
    Mật độ pixel tối (độ sáng < 215) phải chiếm ít nhất 35% diện tích bbox.
    """
    x1, y1, x2, y2 = map(int, bbox)
    h_img, w_img = cv_img.shape[:2]
    x1 = max(0, min(x1, w_img - 1))
    x2 = max(0, min(x2, w_img - 1))
    y1 = max(0, min(y1, h_img - 1))
    y2 = max(0, min(y2, h_img - 1))
    if x2 <= x1 or y2 <= y1:
        return False
    crop = cv_img[y1:y2, x1:x2]
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    fill_pixels = np.sum(gray_crop < 215)
    total_pixels = gray_crop.size
    return (fill_pixels / total_pixels) >= 0.35


# ── Hàm phân tích ảnh và trích xuất POI bằng OCR & OpenCV ────────────

async def extract_pois_from_screenshot(
    screenshot_bytes: bytes,
    bbox: Tuple[float, float, float, float] = None,
    tx: int = None,
    ty: int = None,
    zoom: int = None,
    img_metadata: dict = None,
) -> Tuple[List[dict], bool]:
    """
    Trích xuất danh sách địa điểm (POI) từ ảnh screenshot bằng Tesseract OCR & OpenCV.
    - screenshot_bytes: byte ảnh gốc (định dạng PNG hoặc JPEG).
    - Trả về: (list POIs, outside_district=False).
    """
    if not screenshot_bytes:
        return [], False

    # 1. Lưu ảnh gốc ra file tạm để xử lý bằng PIL/Tesseract/OpenCV trên Windows an toàn
    fd, temp_img_path = tempfile.mkstemp(suffix=".png")
    temp_processed_path = temp_img_path.replace(".png", "_processed.png")
    
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(screenshot_bytes)

        # 2. Tiền xử lý ảnh bằng OpenCV (Adaptive Thresholding + 2x Resize nếu width < 2000)
        cv_img = cv2.imread(temp_img_path)
        if cv_img is None:
            raise ValueError(f"Could not read image from {temp_img_path}")
            
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        h_orig, w_orig = gray.shape[:2]
        
        # Phóng to 2x nếu chiều rộng ảnh < 2000px, giữ nguyên 1x nếu ảnh đã có độ phân giải cao >= 2000px
        upscale_factor = 2 if w_orig < 2000 else 1
        
        if upscale_factor == 2:
            gray_proc = cv2.resize(gray, (w_orig * 2, h_orig * 2), interpolation=cv2.INTER_LANCZOS4)
        else:
            gray_proc = gray
            
        # Sử dụng ngưỡng nhị phân cố định 200 để tách văn bản tối màu khỏi nền sáng Google Maps cực kỳ sắc nét
        _, thresh = cv2.threshold(gray_proc, 200, 255, cv2.THRESH_BINARY)
            
        cv2.imwrite(temp_processed_path, thresh)

        # 3. Gọi Tesseract OCR để lấy cấu trúc TSV với chế độ PSM 11 (Sparse text)
        cmd = ["tesseract", temp_processed_path, "stdout", "-l", "vie+eng", "--psm", "11", "tsv"]
        loop = asyncio.get_event_loop()
        
        # Chạy subprocess trong thread pool để tránh chặn đứng event loop asyncio
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        )
        
        tsv_content = result.stdout if result.returncode == 0 else ""
        if not tsv_content and result.returncode != 0:
            logger.error("Tesseract failed with stderr: %s", result.stderr)

        # 4. Parse nội dung TSV và chia tọa độ về tỷ lệ ban đầu
        words = []
        if tsv_content:
            reader = csv.DictReader(io.StringIO(tsv_content), delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                if row.get('level') == '5':
                    text = row.get('text', '').strip()
                    conf = float(row.get('conf', 0))
                    if text:
                        # Require higher confidence (>= 70) for short words to filter out noise
                        is_valid = False
                        if len(text) <= 3:
                            if conf >= 70:
                                is_valid = True
                        else:
                            if conf >= 45:
                                is_valid = True
                                
                        if is_valid:
                            # Chia cho upscale_factor để quy đổi về ảnh chụp màn hình gốc
                            words.append({
                                'text': text,
                                'left': int(float(row['left']) / float(upscale_factor)),
                                'top': int(float(row['top']) / float(upscale_factor)),
                                'width': int(float(row['width']) / float(upscale_factor)),
                                'height': int(float(row['height']) / float(upscale_factor)),
                                'conf': conf
                            })

        # 5. Gom nhóm các từ nằm cùng một dòng (Spatial Word Grouping)
        lines = []
        if words:
            sorted_words = sorted(words, key=lambda w: w['left'])
            for w in sorted_words:
                added = False
                for line in lines:
                    l_top = line['top']
                    l_bottom = line['top'] + line['height']
                    w_top = w['top']
                    w_bottom = w['top'] + w['height']
                    
                    overlap_y = min(l_bottom, w_bottom) - max(l_top, w_top)
                    min_h = min(line['height'], w['height'])
                    
                    # Nếu độ đè vĩ độ (vertical overlap) >= 45% chiều cao của từ nhỏ hơn
                    if overlap_y >= 0.45 * min_h:
                        l_right = line['left'] + line['width']
                        gap_x = w['left'] - l_right
                        
                        # Khoảng cách ngang nhỏ (trong khoảng 12 pixel để tránh nhập nhèm POI liền kề)
                        if -12 <= gap_x <= 12:
                            line['words'].append(w)
                            new_left = min(line['left'], w['left'])
                            new_top = min(line['top'], w['top'])
                            new_right = max(l_right, w['left'] + w['width'])
                            new_bottom = max(l_bottom, w['top'] + w['height'])
                            
                            line['left'] = new_left
                            line['top'] = new_top
                            line['width'] = new_right - new_left
                            line['height'] = new_bottom - new_top
                            added = True
                            break
                if not added:
                    lines.append({
                        'words': [w],
                        'left': w['left'],
                        'top': w['top'],
                        'width': w['width'],
                        'height': w['height']
                    })

        # Lọc ký tự đặc biệt của dòng và giữ lại các dòng hợp lệ (sử dụng Tesseract text tạm thời trước OCR)
        valid_lines = []
        for line in lines:
            line['words'].sort(key=lambda w: w['left'])
            text = " ".join(w['text'] for w in line['words']).strip()
            avg_conf = sum(w['conf'] for w in line['words']) / len(line['words'])
            
            text = re.sub(r'[^\w\s\d,.\-\(\)\/]', '', text).strip()
            if text and len(text) >= 2:
                line_bbox = [line['left'], line['top'], line['left'] + line['width'], line['top'] + line['height']]
                line_color = _get_region_color(cv_img, line_bbox)
                valid_lines.append({
                    'text': text,
                    'left': line['left'],
                    'top': line['top'],
                    'width': line['width'],
                    'height': line['height'],
                    'conf': avg_conf,
                    'color': line_color
                })

        # 6. Nhận diện các ứng viên Icon trên bản đồ bằng OpenCV (Contour Analysis)
        scale = img_metadata.get("scale", 1.0) if img_metadata else 1.0
        candidate_icons = []
        if cv_img is not None:
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Kích thước icon chuẩn ở zoom 19-21 thường từ 14px đến 35px, nhân với tỷ lệ scale động (mở rộng biên độ để tránh bỏ sót)
            min_icon_size = max(10, int(12 * scale))
            max_icon_size = int(40 * scale)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                cx_box, cy_box, cw_box, ch_box = cv2.boundingRect(c)
                if min_icon_size <= cw_box <= max_icon_size and min_icon_size <= ch_box <= max_icon_size:
                    aspect_ratio = float(cw_box) / ch_box
                    if 0.7 <= aspect_ratio <= 1.4:
                        icon_bbox = [cx_box, cy_box, cx_box + cw_box, cy_box + ch_box]
                        if not is_solid_icon(cv_img, icon_bbox):
                            continue
                        icon_color = _get_region_color(cv_img, icon_bbox)
                        candidate_icons.append({
                            'x': cx_box + cw_box / 2.0,
                            'y': cy_box + ch_box / 2.0,
                            'left': float(cx_box),
                            'top': float(cy_box),
                            'right': float(cx_box + cw_box),
                            'bottom': float(cy_box + ch_box),
                            'w': cw_box,
                            'h': ch_box,
                            'bbox': [float(cx_box), float(cy_box), float(cx_box + cw_box), float(cy_box + ch_box)],
                            'color': icon_color
                        })

        # Lọc bỏ các icon đè/trùng với các chữ phát hiện từ Tesseract (tránh nhận nhầm chữ cái/dấu nháy kép làm icon)
        non_text_icons = []
        for icon in candidate_icons:
            is_text_overlap = False
            for w in words:
                # Bỏ qua kiểm tra đè chữ với các chữ cái đơn lẻ hoặc ký tự đặc biệt hay bị nhận nhầm
                if len(w['text']) <= 2 and w['text'].lower() in {"o", "0", "c", "©", "@", "a", "q", "v", "x", "\"", "'", ",", "."}:
                    continue
                overlap_x = max(0, min(icon['right'], w['left'] + w['width']) - max(icon['left'], w['left']))
                overlap_y = max(0, min(icon['bottom'], w['top'] + w['height']) - max(icon['top'], w['top']))
                if overlap_x > 2 and overlap_y > 2:
                    is_text_overlap = True
                    break
            if not is_text_overlap:
                non_text_icons.append(icon)
        candidate_icons = non_text_icons

        # Khử trùng các icon nằm quá sát nhau (NMS dựa trên kích thước contour)
        deduped_icons = []
        sorted_icons = sorted(candidate_icons, key=lambda i: i['w'] * i['h'], reverse=True)
        for icon in sorted_icons:
            too_close = False
            for selected in deduped_icons:
                dist = ((icon['x'] - selected['x'])**2 + (icon['y'] - selected['y'])**2)**0.5
                if dist <= 18.0 * scale:
                    too_close = True
                    break
            if not too_close:
                deduped_icons.append(icon)
        candidate_icons = deduped_icons

        # 7. Đối sánh các dòng chữ thô với các Icon lân cận (Many-to-One) trước khi chạy VietOCR
        line_matches = {} # l_idx -> (best_i_idx, dist)
        for l_idx, line in enumerate(valid_lines):
            lx = line['left']
            ly = line['top']
            lw = line['width']
            lh = line['height']
            line_center_x = lx + lw / 2.0
            line_center_y = ly + lh / 2.0
            
            for i_idx, icon in enumerate(candidate_icons):
                ix, iy = icon['x'], icon['y']
                
                is_horizontal_adjacent = (
                    (icon['right'] >= lx - OCR_HORIZONTAL_GAP_MAX * scale and 
                     icon['left'] <= lx + OCR_HORIZONTAL_GAP_MAX * scale)
                    or
                    (icon['left'] <= lx + lw + OCR_HORIZONTAL_GAP_MAX * scale and
                     icon['right'] >= lx + lw - OCR_HORIZONTAL_GAP_MAX * scale)
                )
                
                is_vertically_close = abs(iy - line_center_y) <= 45 * scale
                
                if is_horizontal_adjacent and is_vertically_close:
                    # Tính khoảng cách dựa trên cạnh gần nhất của chữ tới tâm icon
                    if ix < lx:
                        dist_x = lx - icon['right']
                    elif ix > lx + lw:
                        dist_x = icon['left'] - (lx + lw)
                    else:
                        dist_x = 0
                    
                    dist_y = abs(iy - line_center_y)
                    dist = (dist_x**2 + dist_y**2)**0.5
                    
                    if l_idx not in line_matches or dist < line_matches[l_idx][1]:
                        line_matches[l_idx] = (i_idx, dist)

        # 7.5 Đối sánh các dòng chưa khớp (unmatched lines) vào cùng icon với dòng đã khớp gần nó (cùng màu, xếp dọc)
        matched_line_indices = set(line_matches.keys())
        unmatched_line_indices = [idx for idx in range(len(valid_lines)) if idx not in matched_line_indices]
        
        for u_idx in unmatched_line_indices:
            u_line = valid_lines[u_idx]
            best_match_idx = None
            min_dist_y = 999999
            
            for m_idx in matched_line_indices:
                m_line = valid_lines[m_idx]
                
                # Khoảng cách dòng dọc gần nhau (không quá 25px)
                dist_y = abs(u_line['top'] - m_line['top'])
                if dist_y > 25 * scale:
                    continue
                    
                # Căn lề trái hoặc lề phải, hoặc có sự đè ngang
                left_aligned = abs(u_line['left'] - m_line['left']) <= 25 * scale
                right_aligned = abs((u_line['left'] + u_line['width']) - (m_line['left'] + m_line['width'])) <= 25 * scale
                horizontal_overlap = max(0, min(u_line['left'] + u_line['width'], m_line['left'] + m_line['width']) - max(u_line['left'], m_line['left'])) > 0
                
                if left_aligned or right_aligned or horizontal_overlap:
                    # Phải cùng tông màu
                    if _colors_are_similar(u_line['color'], m_line['color'], thresh_h=25, thresh_s=75, thresh_v=75):
                        if dist_y < min_dist_y:
                            min_dist_y = dist_y
                            best_match_idx = m_idx
                            
            if best_match_idx is not None:
                # Kế thừa icon của dòng đã khớp
                i_idx, _ = line_matches[best_match_idx]
                line_matches[u_idx] = (i_idx, min_dist_y)
                matched_line_indices.add(u_idx)

        # 8. Gom cụm các dòng theo Icon để phân nhóm chạy VietOCR chọn lọc
        icon_to_lines = {}
        for l_idx, (i_idx, dist) in line_matches.items():
            if i_idx not in icon_to_lines:
                icon_to_lines[i_idx] = []
            icon_to_lines[i_idx].append(l_idx)
            
        _GENERIC_WORDS_SET = {
            "cafe", "coffee", "shop", "store", "restaurant", "hotel", "bank", "market",
            "church", "school", "park", "pharmacy", "clinic", "spa", "gym", "bar",
            "pub", "hostel", "supermarket", "mall", "center", "centre", "tower",
            "building", "office", "station", "post office", "post_office", "landmark",
            "nhà hàng", "quán ăn", "cà phê", "ngân hàng", "khách sạn", "trường học",
            "bệnh viện", "chợ", "công viên", "nhà thờ", "siêu thị", "tòa nhà", "văn phòng",
            "bưu điện", "trụ sở", "cửa hàng", "cửa hiệu", "hiệu thuốc", "quầy thuốc", "tiệm",
            "vegetarian", "vegan", "convenience", "clothing", "apparel", "souvenir", "gift", "gifts",
            "fashion", "beauty", "salon", "bookstore", "atm", "travel", "agency", "service",
            "lịch", "sử", "di", "tích", "dịch", "vụ", "tạp", "hóa", "tiện", "lợi", "bán", "lẻ", "sách",
            "giày", "dép", "quần", "áo", "thời", "trang", "mỹ", "phẩm", "nước", "hoa", "perfume", "costume",
            "century", "historical", "heritage", "museum", "monument", "shrine", "attraction", "tourist",
            "and", "or", "of", "in", "the", "a", "&", "to", "for", "with", "by", "-",
            "nhật", "bản", "hàn", "quốc", "pháp", "mỹ", "việt", "nam", "trung", "quốc", "thái", "lan",
            "vietnamese", "japanese", "korean", "french", "italian", "american", "thai", "western", "asian"
        }

        def is_description_line(text_str: str) -> bool:
            text_lower = text_str.lower().strip()
            if _is_generic_name(text_lower):
                return True
            
            # Tách thành các từ đơn
            words = [w for w in re.split(r'\W+', text_lower) if w]
            if not words:
                return True
                
            # Nếu tất cả các từ trong dòng đều là từ mô tả chung chung (không chứa tên riêng)
            # thì dòng đó là dòng mô tả loại hình/dịch vụ
            if all(w in _GENERIC_WORDS_SET for w in words):
                return True
                
            return False

        # 9. Chỉ chạy VietOCR cho các dòng đã khớp với icon và tạo danh sách POI gộp
        pois = []
        for i_idx, l_indices in icon_to_lines.items():
            icon = candidate_icons[i_idx]
            
            # Sắp xếp các dòng theo thứ tự từ trên xuống dưới
            matched_lines = [valid_lines[idx] for idx in l_indices]
            matched_lines.sort(key=lambda x: x['top'])
            
            # Đồng nhất căn lề trái cho toàn bộ các dòng thuộc cùng một POI
            min_l = min(line['left'] for line in matched_lines)
            
            name_parts = []
            for line in matched_lines:
                lx = min_l
                ly = line['top']
                rx = max(line['left'] + line['width'], lx + 10)
                lw = rx - lx
                lh = line['height']
                
                ix_left = icon['left']
                ix_right = icon['right']
                
                # Loại bỏ vùng chồng lấn với biểu tượng
                if icon['x'] < lx + lw / 2.0:
                    if lx < ix_right:
                        lx = int(ix_right + 2)
                        lw = max(0, int(rx - lx))
                else:
                    if rx > ix_left:
                        lw = max(0, int(ix_left - 2 - lx))
                        
                if lw <= 0:
                    continue
                    
                line_bbox = [lx, ly, lx + lw, ly + lh]
                
                # Chạy VietOCR chọn lọc cho vùng bbox của dòng chữ
                text_read = _recognize_text_crop_vietocr(cv_img, line_bbox, fallback=line['text'])
                text_cleaned = _clean_spelling(text_read)
                
                if is_description_line(text_cleaned):
                    logger.info("  [Desc-Filtered] Bỏ dòng mô tả: '%s'", text_cleaned)
                    continue
                name_parts.append(text_cleaned)
                
            # Bỏ qua POI hoàn toàn nếu không có dòng tên hợp lệ
            if not name_parts:
                logger.info("  [POI-Filtered] Bỏ POI tại (%f, %f) vì không có dòng tên hợp lệ.", icon['x'], icon['y'])
                continue
                
            full_name = " ".join(name_parts)
            full_name = _clean_spelling(full_name)
            
            # Khử nhiễu ký tự đơn lẻ ngoại trừ chữ số và một số từ đơn đặc trưng
            words_in_name = full_name.split()
            cleaned_words = []
            for w in words_in_name:
                if len(w) > 1:
                    cleaned_words.append(w)
                elif w.isdigit() or w.lower() in {'a', 'i'}:
                    cleaned_words.append(w)
            full_name = " ".join(cleaned_words)
            
            full_name = re.sub(r'\s+', ' ', full_name).strip()
            full_name = _clean_spelling(full_name)
            
            if len(full_name) < 2 or not any(c.isalpha() for c in full_name):
                continue
            
            # Độ tin cậy trung bình
            avg_conf = sum(line['conf'] for line in matched_lines) / len(matched_lines)
            
            # Tính toán bbox bao phủ toàn bộ các dòng được gộp
            min_l = min(line['left'] for line in matched_lines)
            min_t = min(line['top'] for line in matched_lines)
            max_r = max(line['left'] + line['width'] for line in matched_lines)
            max_b = max(line['top'] + line['height'] for line in matched_lines)
            
            pois.append({
                "name": full_name,
                "x": icon['x'], # Tọa độ GPS tâm POI chính là tâm của biểu tượng (icon)
                "y": icon['y'],
                "confidence": avg_conf / 100.0,
                "bbox": [float(min_l), float(min_t), float(max_r - min_l), float(max_b - min_t)],
                "has_icon": True
            })

        global _last_candidate_icons, _last_valid_lines, _last_line_to_icon, _last_potential_line_matches
        _last_candidate_icons = candidate_icons
        _last_valid_lines = valid_lines
        _last_line_to_icon = line_matches
        _last_potential_line_matches = []

        logger.info("OCR Vision Done: %d POIs extracted (Confirmed 100%% Local OCR)", len(pois))
        return pois, False

    except Exception as exc:
        logger.error("OCR Extraction Pipeline error: %s", exc, exc_info=exc)
        return [], False

    finally:
        # 9. Dọn dẹp file tạm trên ổ đĩa để giải phóng dung lượng
        for path in [temp_img_path, temp_processed_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass


def remove_background(image_bytes: bytes) -> bytes:
    """
    Khử nền cho ảnh chụp màn hình bản đồ để làm rõ các đoạn text và icon.
    Chuyển ảnh về grayscale, tạo mask cho các pixel cực sáng (> 210) và đổi chúng sang màu trắng.
    Đồng thời làm đậm các nét chữ/icon tối màu (<= 170) mà không làm nổi bật nền đường xám nhạt.
    Trả về dữ liệu bytes của ảnh đã khử nền dưới dạng PNG.
    """
    try:
        # Decode bytes to OpenCV image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return image_bytes

        # Chuyển sang ảnh xám để tìm nền cực sáng (> 210)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray > 210
        enhance_mask = gray <= 170

        # Chuyển sang HSV để tăng độ rực màu (S) và giảm độ sáng (V) giúp chữ đậm hơn
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # 1. Chỉ làm đậm các pixel thực sự tối (ví dụ: nét chữ, icon có gray <= 170)
        v_fg = v[enhance_mask].astype(np.float32) * 0.65
        v[enhance_mask] = np.clip(v_fg, 0, 255).astype(np.uint8)

        # 2. Chỉ tăng độ rực màu của các pixel này
        s_fg = s[enhance_mask].astype(np.float32) * 1.5
        s[enhance_mask] = np.clip(s_fg, 0, 255).astype(np.uint8)

        # Ghép lại thành ảnh BGR
        enhanced_hsv = cv2.merge([h, s, v])
        processed_img = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

        # Đặt các pixel thuộc nền cực sáng thành màu trắng tinh (255, 255, 255)
        processed_img[mask] = [255, 255, 255]

        # Encode lại sang PNG bytes
        success, encoded_img = cv2.imencode('.png', processed_img)
        if success:
            return encoded_img.tobytes()
    except Exception as e:
        logger.warning("Lỗi khi khử nền và tăng nét ảnh: %s", e)
    return image_bytes


def draw_detections(image_bytes: bytes, pois: List[dict], crop_x: int = 0, crop_y: int = 0) -> bytes:
    """
    Vẽ khung chữ nhật (bbox) và nhãn văn bản (name) của từng địa điểm đã nhận diện
    lên ảnh nền đã được khử. Hỗ trợ Unicode tiếng Việt bằng PIL.
    - image_bytes: bytes của ảnh nền đã khử (PNG/JPEG)
    - pois: danh sách các POI từ extract_pois_from_screenshot
    - crop_x, crop_y: offset cắt của ảnh lưu so với ảnh chụp full
    """
    try:
        import cv2
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        # Decode BGR image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return image_bytes

        # Vẽ các khung chữ nhật màu đỏ lên ảnh OpenCV trước
        for poi in pois:
            bbox = poi.get("bbox")
            if not bbox:
                continue
            left, top, w, h = bbox
            x1 = int(left - crop_x)
            y1 = int(top - crop_y)
            x2 = int(x1 + w)
            y2 = int(y1 + h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Chuyển sang ảnh PIL để vẽ Unicode tiếng Việt
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # Load font chữ hỗ trợ Unicode tiếng Việt
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except Exception:
            try:
                font = ImageFont.load_default(size=15)
            except TypeError:
                font = ImageFont.load_default()

        for poi in pois:
            bbox = poi.get("bbox")
            name = poi.get("name", "")
            if not bbox or not name:
                continue
            
            left, top, w, h = bbox
            x1 = int(left - crop_x)
            y1 = int(top - crop_y)

            # Tính toán kích thước chữ để vẽ nền nhãn
            try:
                bbox_t = draw.textbbox((x1, y1), name, font=font)
                label_w = bbox_t[2] - bbox_t[0]
                label_h = bbox_t[3] - bbox_t[1]
            except Exception:
                label_w = len(name) * 8
                label_h = 15
                bbox_t = [x1, y1 - label_h - 2, x1 + label_w, y1]

            # Xác định vị trí vẽ nhãn chữ theo trục Y
            ty = y1 - label_h - 6
            if ty < 0:
                ty = y1 + h + 4

            # Vẽ nền màu xanh nhạt (RGB: 230, 230, 255)
            draw.rectangle(
                [x1, ty - 2, x1 + label_w + 4, ty + label_h + 4],
                fill=(230, 230, 255)
            )
            # Viết tên POI
            draw.text((x1 + 2, ty), name, fill=(0, 0, 0), font=font)

        # Chuyển ngược về ảnh OpenCV BGR
        enhanced_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Encode lại thành PNG bytes
        success, encoded_img = cv2.imencode('.png', enhanced_img)
        if success:
            return encoded_img.tobytes()
    except Exception as e:
        logger.warning("Lỗi khi vẽ nét nhận diện Unicode: %s", e)
    return image_bytes


