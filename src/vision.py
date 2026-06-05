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
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w_img, x2 + pad_x)
    y2 = min(h_img, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return fallback.strip()

    crop = cv_img[y1:y2, x1:x2]
    if crop.size == 0:
        return fallback.strip()

    try:
        # VietOCR đọc tốt hơn khi crop chữ nhỏ được phóng nhẹ.
        ch, cw = crop.shape[:2]
        scale = 2 if max(ch, cw) < 220 else 1
        if scale > 1:
            crop = cv2.resize(crop, (cw * scale, ch * scale), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        text = predictor.predict(pil_img)
        text = re.sub(r'\s+', ' ', (text or '')).strip()
        if len(text) >= 2:
            logger.debug("[VietOCR] '%s' -> '%s'", fallback, text)
            return text
    except Exception as exc:
        logger.debug("[VietOCR] Crop OCR lỗi: %s", exc)
    return fallback.strip()


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
            
        # Sử dụng ngưỡng nhị phân cố định 180 để tách văn bản tối màu khỏi nền sáng OSM cực kỳ sắc nét
        _, thresh = cv2.threshold(gray_proc, 180, 255, cv2.THRESH_BINARY)
            
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
            reader = csv.DictReader(io.StringIO(tsv_content), delimiter='\t')
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

        # Lọc ký tự đặc biệt của dòng và giữ lại các dòng hợp lệ
        valid_lines = []
        for line in lines:
            line['words'].sort(key=lambda w: w['left'])
            text = " ".join(w['text'] for w in line['words']).strip()
            avg_conf = sum(w['conf'] for w in line['words']) / len(line['words'])
            
            # Xóa các ký tự nhiễu nhưng giữ dấu tiếng Việt
            text = re.sub(r'[^\w\s\d,.\-\(\)\/]', '', text).strip()
            if text and len(text) >= 2:
                # Trích xuất màu sắc đại diện cho dòng chữ này
                line_bbox = [line['left'], line['top'], line['left'] + line['width'], line['top'] + line['height']]
                text = _recognize_text_crop_vietocr(cv_img, line_bbox, fallback=text)
                text = re.sub(r'[^\w\s\d,.\-\(\)\/]', '', text).strip()
                if not text or len(text) < 2:
                    continue
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

        # 6. Gom cụm các dòng xếp chồng (Spatial Line Clustering) thành nhãn POI đa dòng
        # Sắp xếp các dòng từ trên xuống dưới
        valid_lines.sort(key=lambda l: l['top'])
        labels = []
        
        for line in valid_lines:
            merged = False
            for cluster in labels:
                c_bottom = cluster['top'] + cluster['height']
                c_left = cluster['left']
                c_right = cluster['left'] + cluster['width']
                
                v_gap = line['top'] - c_bottom
                # Hạn chế chiều cao merging gap chặt chẽ
                max_v_gap = 0.7 * min(line['height'], cluster['height']) + 1
                min_v_gap = -1.2 * min(line['height'], cluster['height'])
                
                # Kiểm tra xem khoảng cách dọc có nằm trong khoảng cho phép không
                if min_v_gap <= v_gap <= max_v_gap:
                    # Căn chỉnh tâm ngang hợp lý (cho phép lệch tối đa 35px đối với ảnh 2x)
                    c_center_x = (c_left + c_right) / 2
                    line_center_x = line['left'] + line['width'] / 2
                    x_diff = abs(c_center_x - line_center_x)
                    
                    # Hoặc có độ đè ngang từ 35% trở lên
                    h_overlap = min(c_right, line['left'] + line['width']) - max(c_left, line['left'])
                    h_overlap_ratio = h_overlap / min(cluster['width'], line['width']) if min(cluster['width'], line['width']) > 0 else 0
                    
                    if x_diff <= 35 or h_overlap_ratio >= 0.35:
                        cluster['text'] += " " + line['text']
                        new_left = min(c_left, line['left'])
                        new_top = min(cluster['top'], line['top'])
                        new_right = max(c_right, line['left'] + line['width'])
                        new_bottom = max(c_bottom, line['top'] + line['height'])
                        
                        cluster['left'] = new_left
                        cluster['top'] = new_top
                        cluster['width'] = new_right - new_left
                        cluster['height'] = new_bottom - new_top
                        cluster['conf'] = (cluster['conf'] + line['conf']) / 2
                        merged = True
                        break
            
            if not merged:
                labels.append({
                    'text': line['text'],
                    'left': line['left'],
                    'top': line['top'],
                    'width': line['width'],
                    'height': line['height'],
                    'conf': line['conf'],
                    'color': line['color']
                })

        # Helper hàm xác định nhãn POI có hợp lệ không (tránh rác chữ ngắn < 4 ký tự)
        def is_valid_poi_name(name_str: str) -> bool:
            c_name = name_str.strip()
            if not c_name:
                return False
                
            # Bỏ qua các địa chỉ/số nhà thuần số hoặc dạng số nhà (VD: "15", "15A", "12/4", "108")
            if re.match(r'^\d+$', c_name) or re.match(r'^\d+[a-zA-Z]$', c_name) or re.match(r'^\d+(/\d+)+[a-zA-Z]?$', c_name):
                return False

            # Bỏ qua các nhãn hành chính như Phường, Quận, Đường (nếu lọt qua bộ lọc đường)
            # Ví dụ: "Quận 1", "Q. 1", "Q.1", "Phường Bến Nghé", "P. Bến Nghé", "P.Bến Nghé", "P. 15", "Phường 15"
            c_name_lower = c_name.lower()
            admin_patterns = [
                r'^quận\s+\d+$', r'^q\.\s*\d+$',
                r'^phường\s+.*$', r'^p\.\s*.*$',
                r'^district\s+\d+$', r'^ward\s+\d+$'
            ]
            for pat in admin_patterns:
                if re.match(pat, c_name_lower):
                    return False

            # Bỏ qua các tên quốc gia đứng riêng lẻ (thường do tách dòng từ Tổng Lãnh sự quán)
            if c_name_lower in _COUNTRY_NAMES:
                return False

            # Nếu tên < 4 ký tự, có khả năng cao là rác trừ khi:
            # 1. Viết hoa hoàn toàn chỉ chứa chữ cái từ A-Z (viết tắt như KFC, ATM)
            # 2. Là một số từ tiếng Việt ngắn phổ biến có nghĩa trên bản đồ
            if len(c_name) < 4:
                if re.match(r'^[A-Z]+$', c_name):
                    return True
                no_acc = _strip_vietnamese_accents(c_name)
                if no_acc in {"pho", "cho", "cau", "bun", "che", "ga", "kho", "rap", "dinh", "com", "kem"}:
                    return True
                return False
            return True

        # 7. Nhận diện các ứng viên Icon trên bản đồ bằng OpenCV (Contour Analysis)
        # Sử dụng thông số scale từ metadata để thích ứng với DPI của từng màn hình
        scale = img_metadata.get("scale", 1.0) if img_metadata else 1.0
        candidate_icons = []
        # cv_img đã được đọc ở trên trong phần tiền xử lý
        if cv_img is not None:
            # gray đã được convert ở trên
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Kích thước icon chuẩn ở zoom 19 thường từ 10px đến 28px, nhân với tỷ lệ scale động
            min_icon_size = max(6, int(8 * scale))
            max_icon_size = int(35 * scale)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                cx_box, cy_box, cw_box, ch_box = cv2.boundingRect(c)
                if min_icon_size <= cw_box <= max_icon_size and min_icon_size <= ch_box <= max_icon_size:
                    aspect_ratio = float(cw_box) / ch_box
                    # Tỷ lệ khung hình vuông vắn (từ 0.7 đến 1.4)
                    if 0.7 <= aspect_ratio <= 1.4:
                        icon_bbox = [cx_box, cy_box, cx_box + cw_box, cy_box + ch_box]
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

        # 8. So khớp nhãn chữ với các Icon lân cận và gán tọa độ thích hợp
        # Lọc danh sách nhãn hợp lệ trước
        valid_labels = []
        for label in labels:
            label['text'] = _clean_spelling(label['text'])
            if _is_generic_name(label['text']) or not is_valid_poi_name(label['text']) or _is_street_name(label['text']):
                logger.info("  [OCR-Filter] Bỏ nhãn không hợp lệ/generic/đường phố: '%s'", label['text'])
                continue
            valid_labels.append(label)
            
        # Sắp xếp các nhãn theo thứ tự đọc (từ trên xuống dưới, cùng dòng thì từ trái sang phải)
        if valid_labels:
            sorted_by_top = sorted(valid_labels, key=lambda x: x['top'])
            rows = []
            current_row = []
            for item in sorted_by_top:
                if not current_row:
                    current_row.append(item)
                else:
                    avg_top = sum(x['top'] for x in current_row) / len(current_row)
                    # Độ lệch dòng tối đa 15px để coi là cùng dòng ngang
                    if abs(item['top'] - avg_top) <= 15:
                        current_row.append(item)
                    else:
                        current_row.sort(key=lambda x: x['left'])
                        rows.extend(current_row)
                        current_row = [item]
            if current_row:
                current_row.sort(key=lambda x: x['left'])
                rows.extend(current_row)
            valid_labels = rows
            
        # Tìm tất cả các cặp khớp tiềm năng (label, icon) thỏa mãn điều kiện hình học và màu sắc
        potential_matches = []
        for l_idx, label in enumerate(valid_labels):
            lx = label['left']
            ly = label['top']
            lw = label['width']
            lh = label['height']
            label_center_x = lx + lw / 2.0
            label_center_y = ly + lh / 2.0
            
            for i_idx, icon in enumerate(candidate_icons):
                ix, iy = icon['x'], icon['y']
                
                # Kiểm tra icon nằm ngang (trái hoặc phải) của text label
                # User confirmed: Icon CÓ THỂ TRÁI hoặc PHẢI, khoảng cách ~5px, nằm ngang nhau
                is_horizontal_adjacent = (
                    # Y-axis alignment: Icon Y gần text center Y (trong 70% chiều cao text)
                    # Text có thể cao hơn nếu nhiều dòng, nên dùng label_center_y
                    abs(iy - label_center_y) <= lh * OCR_ICON_Y_ALIGN_RATIO
                    and
                    # Horizontal proximity: Icon gần text theo phương ngang (trái HOẶC phải)
                    (
                        # Case 1: Icon ở BÊN TRÁI text (icon.right gần label.left)
                        (icon['right'] >= lx - OCR_HORIZONTAL_GAP_MAX * scale and 
                         icon['left'] <= lx + OCR_HORIZONTAL_GAP_MAX * scale)
                        or
                        # Case 2: Icon ở BÊN PHẢI text (icon.left gần label.right)
                        (icon['left'] <= lx + lw + OCR_HORIZONTAL_GAP_MAX * scale and
                         icon['right'] >= lx + lw - OCR_HORIZONTAL_GAP_MAX * scale)
                    )
                )
                
                if is_horizontal_adjacent:
                    # Kiểm tra màu sắc của nhãn chữ và biểu tượng tương đồng nhau
                    if _colors_are_similar(label['color'], icon['color'], thresh_h=25, thresh_s=75, thresh_v=75):
                        dist = ((ix - label_center_x) ** 2 + (iy - label_center_y) ** 2) ** 0.5
                        potential_matches.append((dist, l_idx, i_idx))
                        
        # Sắp xếp các cặp theo khoảng cách tăng dần để ưu tiên cặp gần nhất
        potential_matches.sort(key=lambda x: x[0])
        
        matched_labels = {}  # l_idx -> i_idx
        matched_icons = set()  # set of i_idx
        
        for dist, l_idx, i_idx in potential_matches:
            if l_idx not in matched_labels and i_idx not in matched_icons:
                matched_labels[l_idx] = i_idx
                matched_icons.add(i_idx)
                
        pois = []
        for l_idx, label in enumerate(valid_labels):
            lx = label['left']
            ly = label['top']
            lw = label['width']
            lh = label['height']
            label_center_x = lx + lw / 2.0
            label_center_y = ly + lh / 2.0
            
            if l_idx in matched_labels:
                icon = candidate_icons[matched_labels[l_idx]]
                # Pixel exact invariant: có icon -> bắt buộc dùng tâm bbox icon.
                poi_x = (icon['left'] + icon['right']) / 2.0
                poi_y = (icon['top'] + icon['bottom']) / 2.0
                has_icon = True
                logger.info(
                    "  [PixelExact-Icon] '%s' bbox_icon=%s center=(%.1f, %.1f)",
                    label['text'], icon['bbox'], poi_x, poi_y
                )
            else:
                # Pixel exact invariant: không icon -> bắt buộc dùng tâm bbox label đã merge.
                poi_x = (lx + lx + lw) / 2.0
                poi_y = (ly + ly + lh) / 2.0
                has_icon = False
                logger.info(
                    "  [PixelExact-Text] '%s' bbox_text=%s center=(%.1f, %.1f)",
                    label['text'], [float(lx), float(ly), float(lx + lw), float(ly + lh)], poi_x, poi_y
                )
                
            pois.append({
                "name": label['text'],
                "x": poi_x,
                "y": poi_y,
                "confidence": label['conf'] / 100.0,
                "bbox": [float(lx), float(ly), float(lx + lw), float(ly + lh)],
                "has_icon": has_icon
            })

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
