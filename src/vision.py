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
        DETECT_ENHANCE_ENABLED,
        DETECT_COLOR_THRESH_H,
        DETECT_COLOR_THRESH_S,
        DETECT_COLOR_THRESH_V,
        DETECT_HORIZONTAL_GAP_MAX,
        DETECT_VERTICAL_BELOW_GAP_MAX,
        DETECT_POI_BOX_PAD_PX,
        DETECT_CONTEXT_EXPAND_ENABLED,
        DETECT_CONTEXT_VERTICAL_GAP_MAX,
        DETECT_CONTEXT_X_ALIGN_MAX,
    )
except Exception:
    OCR_ENGINE = "tesseract"
    VIETOCR_MODEL = "vgg_transformer"
    VIETOCR_DEVICE = "cpu"
    OCR_TEXT_PAD_PX = 4
    OCR_HORIZONTAL_GAP_MAX = 10
    OCR_ICON_Y_ALIGN_RATIO = 0.7
    DETECT_ENHANCE_ENABLED = True
    DETECT_COLOR_THRESH_H = 25
    DETECT_COLOR_THRESH_S = 70
    DETECT_COLOR_THRESH_V = 70
    DETECT_HORIZONTAL_GAP_MAX = 18
    DETECT_VERTICAL_BELOW_GAP_MAX = 28
    DETECT_POI_BOX_PAD_PX = 6
    DETECT_CONTEXT_EXPAND_ENABLED = True
    DETECT_CONTEXT_VERTICAL_GAP_MAX = 58
    DETECT_CONTEXT_X_ALIGN_MAX = 70

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
    "nhà hàng",    "nhà thờ", "siêu thị", "tòa nhà", "văn phòng", "bưu điện", "trụ sở", "cửa hàng",
    "cửa hiệu", "hiệu thuốc", "quầy thuốc", "nhà hát", "rạp xiếc", "bảo tàng", "di tích",
    "lăng tẩm", "lăng", "đền", "miếu", "đình", "chùa", "tượng đài", "đài kỷ niệm",
    "dinh", "dinh thự", "phủ", "biệt thự cổ", "nhà cổ", "di sản",
    # Mở rộng các từ loại hình tiếng Việt/Anh chung chung khác
    "rau sạch", "trái cây", "tạp hóa", "quán nước", "trà sữa", "ăn vặt", "bánh mì",
    "cửa hàng tiện lợi", "siêu thị mini", "atm", "rạp chiếu phim", "nhà sách",
    "hiệu sách", "lịch", "lịch sử", "địa điểm", "bản đồ", "tòa đại sứ",
    "lãnh sự", "lãnh sự quán", "đại sứ quán", "tổng lãnh sự quán", "ủy ban",
    "ủy ban nhân dân", "ubnd", "trụ sở ubnd", "công an", "đồn công an",
    "trạm y tế", "nhà khách", "nhà nghỉ", "biệt thự", "chung cư",
    # Thêm các từ rác/chung chung khi đứng độc lập (thường do tách dòng)
    "soon", "coming soon", "open soon", "tương", "tượng", "hoa binh", "hòa bình",
    "phong cách", "thế kỷ", "century", "style", "architecture", "kiến trúc"
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


def _is_low_saturation(color: Optional[np.ndarray], thresh_s: int = 40) -> bool:
    """
    Kiểm tra xem một màu có độ bão hòa thấp (gần với tông xám/trắng/đen) hay không.
    """
    if color is None:
        return True
    hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0]
    return hsv[1] < thresh_s


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


def _is_street_name(name: str, color: Optional[np.ndarray] = None) -> bool:
    """
    Nhận dạng tên đường bằng các biểu thức chính quy và quy tắc tiếng Việt.
    
    Keyword-based checks (từ khóa giao thông, số hiệu đường) chỉ trả về True 
    nếu CÙNG LÚC:
    1. Màu sắc có độ bão hòa thấp (low saturation - xám/trắng tone)
    2. Độ sáng cao (brightness >= 180) - đặc trưng của chữ TRẮNG
    
    Điều này tránh lọc bỏ POI label có chữ TỐI như "Cổng Đường Sách".
    Heuristic person-name checks (Nguyễn Huệ, Pasteur, v.v.) được phép.
    """
    # Nếu có màu sắc rực rỡ (độ bão hòa cao), chắc chắn không phải tên đường thông thường
    if color is not None and not _is_low_saturation(color, thresh_s=50):
        return False

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

    # Kiểm tra độ sáng của màu sắc để gate keyword-based checks
    # Chỉ áp dụng keyword filter nếu màu sắc là trắng/sáng (brightness >= 180)
    is_bright_color = True
    if color is not None:
        hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0]
        brightness = hsv[2]  # V channel
        is_bright_color = brightness >= 180

    check_str = _FALSE_POSITIVE_PHRASES.sub("", name_lower).strip()
    
    # Chỉ áp dụng keyword filter nếu độ bão hòa thấp AND độ sáng cao
    if is_bright_color and _STREET_KEYWORDS.search(check_str):
        return True

    no_accent = _strip_vietnamese_accents(check_str)
    
    # Chỉ áp dụng numbered road filter nếu độ sáng cao
    if is_bright_color and _NUMBERED_ROAD.match(no_accent.strip()):
        return True

    # ④ Heuristic nhận dạng tên đường mang tên người Việt Nam/nước ngoài:
    # 2 đến 5 từ đơn, không chứa từ chỉ loại hình kinh doanh/dịch vụ.
    # (Heuristic này được phép dù là dark text vì dựa vào mô hình tên người)
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


def _is_likely_place_name(text: str) -> bool:
    """
    Kiểm tra xem text có phải là tên địa điểm/landmark không dựa vào từ khóa đặc trưng.
    Trả về True nếu text chứa các từ khóa địa điểm, giúp tránh lọc nhầm tên địa điểm làm tên đường.
    """
    text_lower = text.lower().strip()
    
    # Các từ khóa đầu tiên hoặc trong tên địa điểm (landmark indicators)
    _PLACE_KEYWORDS = {
        # Công trình kiến trúc
        "cổng", "tượng", "đài", "tháp", "dinh", "phủ", "lầu",
        # Cơ sở tôn giáo
        "nhà thờ", "chùa", "đền", "miếu", "tu viện", "thánh đường",
        # Cơ sở văn hóa
        "bảo tàng", "bưu điện", "thư viện", "rạp", "nhà hát",
        # Thương mại & dịch vụ
        "chợ", "siêu thị", "trung tâm", "plaza", "mall", "market",
        # Y tế & giáo dục
        "bệnh viện", "phòng khám", "trường", "đại học", "học viện",
        # Địa điểm công cộng
        "công viên", "vườn", "park", "garden", "quảng trường", "square",
        # Cơ quan
        "đại sứ quán", "lãnh sự quán", "toà án", "ủy ban",
        # Khác
        "bến", "cảng", "ga", "sân bay", "nhà ga", "trạm",
        "khách sạn", "hotel", "hostel", "cafe", "coffee", "restaurant",
        "circle k", "family mart", "vinmart", "co.op",
        "di tích", "lăng", "tượng đài", "dinh", "phủ", "đài", "tháp",
    }
    
    # Kiểm tra xem text có chứa bất kỳ từ khóa địa điểm nào không
    for keyword in _PLACE_KEYWORDS:
        if keyword in text_lower:
            return True
    
    return False


def _is_slanted_road_name(cv_img: np.ndarray, bbox: Tuple[float, float, float, float]) -> bool:
    """
    Nhận diện đoạn text màu trắng nằm xéo so với màn hình
    và nằm trên nền màu xám (tên đường) bằng phép phân tích mô-men đối xứng (square crop).
    """
    try:
        x1, y1, x2, y2 = map(int, bbox)
        h_img, w_img = cv_img.shape[:2]
        
        # Tính toán tọa độ tâm của bounding box
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        # Tạo vùng crop hình vuông đối xứng dựa trên cạnh dài nhất của bbox để tránh thiên lệch tỷ lệ
        max_side = max(x2 - x1, y2 - y1)
        size = max_side + 16 # Padding thêm 8px mỗi bên
        
        x1_sq = max(0, cx - size // 2)
        y1_sq = max(0, cy - size // 2)
        x2_sq = min(w_img, cx + size // 2)
        y2_sq = min(h_img, cy + size // 2)
        
        if x2_sq <= x1_sq or y2_sq <= y1_sq:
            return False

        crop = cv_img[y1_sq:y2_sq, x1_sq:x2_sq]
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # 1. Trích xuất pixel sáng (chữ trắng)
        bright_mask = gray_crop > 215
        pts = np.argwhere(bright_mask)

        # Số lượng pixel trắng tối thiểu và tối đa
        if len(pts) < 8 or len(pts) > 0.7 * gray_crop.size:
            return False

        # 2. Tính toán độ nghiêng thông qua moments trên crop hình vuông
        m = cv2.moments(bright_mask.astype(np.uint8))
        mu20 = m['mu20']
        mu02 = m['mu02']
        mu11 = m['mu11']

        if abs(mu20 - mu02) > 1e-5 or abs(mu11) > 1e-5:
            theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
            angle_deg = np.degrees(theta) % 180
        else:
            angle_deg = 0.0

        dist_to_horiz = min(angle_deg, 180 - angle_deg)
        dist_to_vert = abs(angle_deg - 90)

        is_slanted = (dist_to_horiz >= 10) and (dist_to_vert >= 10)
        if not is_slanted:
            return False

        # 3. Kiểm tra nền xung quanh có phải nền bản đồ sáng (xám/xanh xám) hay không
        bg_mask = (gray_crop >= 150) & (gray_crop <= 220)
        if np.sum(bg_mask) > 5:
            mean_val = np.mean(gray_crop[bg_mask])
            is_grey_bg = (mean_val >= 160)
        else:
            is_grey_bg = False

        return bool(is_slanted and is_grey_bg)
    except Exception as e:
        logger.debug("Error in _is_slanted_road_name: %s", e)
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
        r"\b[đĐ]ại\s+[sS]ự\s+[qQ]uản\b": "Đại sứ quán",
        
        # Dinh Độc Lập
        r"\b[dD]inh\s+[đĐ][ôo]c\s+[lL][âa]p\b": "Dinh Độc Lập",
        r"\b[dD]inh\s+[đĐ]ộc\s+[lL]ập\b": "Dinh Độc Lập",
        
        # Lăng Lê Văn Duyệt / Lăng Ông
        r"\b[lL][ăa]ng\s+[ôO]ng\b": "Lăng Ông",
        r"\b[lL][ăa]ng\s+[lL][êe]\s+[vV][ăa]n\s+[dD]uy[ệẹê]t\b": "Lăng Lê Văn Duyệt",
        
        # Các di tích khác
        r"\b[đĐ][êe]n\s+th[ờo]\b": "Đền thờ",
        r"\b[mM]i[êe]u\b": "Miếu",
        
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


def _recognize_text_crop_vietocr(cv_img: np.ndarray, bbox: Tuple[float, float, float, float], fallback: str = "", icon_bbox: Optional[List[float]] = None) -> str:
    """OCR lại crop chữ bằng VietOCR; fallback về text detector nếu cần."""
    predictor = _get_vietocr_predictor()
    if predictor is None or cv_img is None:
        return fallback.strip()

    h_img, w_img = cv_img.shape[:2]
    # Nới rộng padding crop: pad_x = 10, pad_y = 6
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
        # 1. Khử nhiễu nhưng giữ sắc nét cạnh chữ (Bilateral Filter)
        crop_blur = cv2.bilateralFilter(crop, 5, 65, 65)
        crop_gray = cv2.cvtColor(crop_blur, cv2.COLOR_BGR2GRAY)
        
        # 2. Tăng cường tương phản cục bộ (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        crop_gray = clahe.apply(crop_gray)
        
        # 3. NGƯỠNG ĐỘNG (Otsu): Tự động tìm ngưỡng tối ưu giữa nền và chữ
        # Giúp xử lý tốt cả chữ nhạt màu và chữ đậm trên nhiều loại nền bản đồ.
        _, crop_thresh = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 4. Xóa icon (nếu dính vào vùng crop này) để tránh làm nhiễu VietOCR
        if icon_bbox:
            il, it, ir, ib = icon_bbox
            # Chuyển tọa độ icon sang hệ tọa độ của crop_thresh
            rx1 = int(il - x1_pad)
            ry1 = int(it - y1_pad)
            rx2 = int(ir - x1_pad)
            ry2 = int(ib - y1_pad)
            
            ch, cw = crop_thresh.shape[:2]
            rx1 = max(0, min(rx1, cw - 1))
            ry1 = max(0, min(ry1, ch - 1))
            rx2 = max(0, min(rx2, cw - 1))
            ry2 = max(0, min(ry2, ch - 1))
            
            # Chỉ xóa nếu icon không quá to (tránh xóa nhầm cả cụm text)
            if rx2 > rx1 and ry2 > ry1 and (rx2-rx1) < cw * 0.8:
                crop_thresh[ry1:ry2, rx1:rx2] = 255

        # 4. Tạo mặt nạ mask để chỉ lấy đúng vùng text của dòng này (có padding an toàn)
        lx1 = x1 - x1_pad
        ly1 = y1 - y1_pad
        lx2 = x2 - x1_pad
        ly2 = y2 - y1_pad
        
        mask = np.zeros(crop_thresh.shape, dtype=np.uint8)
        # Thêm padding 3px cho mask để không cắt phạm vào nét chữ/dấu
        ml1 = max(0, ly1 - 3)
        ml2 = min(ly2 + 3, crop_thresh.shape[0])
        mc1 = max(0, lx1 - 3)
        mc2 = min(lx2 + 3, crop_thresh.shape[1])
        mask[ml1:ml2, mc1:mc2] = 255
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
    Kiểm tra vùng bbox có đủ tín hiệu giống icon POI không.
    Nới hơn rule cũ để không bỏ sót icon nhỏ/mảnh/xám của Google Maps.
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
    if crop.size == 0:
        return False

    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    total_pixels = max(1, gray_crop.size)

    dark_ratio = float(np.sum(gray_crop < 215)) / total_pixels
    edge_ratio = float(np.sum(cv2.Canny(gray_crop, 40, 120) > 0)) / total_pixels
    sat_ratio = float(np.sum((hsv_crop[:, :, 1] > 45) & (hsv_crop[:, :, 2] > 80))) / total_pixels
    mid_gray_ratio = float(np.sum((gray_crop >= 80) & (gray_crop <= 210))) / total_pixels

    # Icon POI có thể là khối màu, outline mảnh, hoặc pin xám nhỏ.
    return (
        dark_ratio >= 0.18
        or sat_ratio >= 0.10
        or (edge_ratio >= 0.08 and mid_gray_ratio >= 0.12)
    )


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
        
        # Phóng to 2x nếu chiều rộng ảnh < 2000px
        upscale_factor = 2 if w_orig < 2000 else 1
        
        if upscale_factor == 2:
            gray_proc = cv2.resize(gray, (w_orig * 2, h_orig * 2), interpolation=cv2.INTER_LANCZOS4)
        else:
            gray_proc = gray
            
        # Vì ảnh đã được khử nền (nền trắng tinh 255), ta sử dụng trực tiếp ảnh xám 
        # để Tesseract nhận diện tốt hơn, tránh mất nét do threshold thêm một lần.
        cv2.imwrite(temp_processed_path, gray_proc)

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
                        
                        # Khoảng cách ngang nhỏ (trong khoảng 18 pixel để tránh tách dòng do kerning/font Google Maps)
                        if -12 <= gap_x <= 18:
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

                # Kiểm tra lọc tên đường: CHỈ khử khi có đủ CẢ 3 yếu tố:
                # 1. Nằm xéo (slanted)
                # 2. Chữ trắng (bright text)
                # 3. Nằm trên nền màu xám (gray background)
                # NHƯNG vẫn giữ lại nếu text chứa từ khóa tên địa điểm (landmark)
                # để tránh khử nhầm tên địa điểm lỡ nằm trên/đè lên đường.
                # if _is_slanted_road_name(cv_img, line_bbox):
                #     if _is_likely_place_name(text):
                #         logger.info("  [Road-Name-Kept] Giữ lại địa điểm dù nằm xéo: '%s'", text)
                #     else:
                #         logger.info("  [Road-Name-Filtered] Bỏ tên đường xéo: '%s' tại bbox %s", text, line_bbox)
                #         continue
                
                # ĐÃ TẮT: Kiểm tra từ khóa tên đường độc lập (quá tay, lọc nhầm tên địa điểm)
                # Ví dụ: "Cổng Đường Sách" bị lọc nhầm vì có từ "Đường"
                # Giờ chỉ dựa vào phát hiện text xéo + trắng + nền xám ở trên
                # if _is_street_name(text, line_color):
                #     logger.info("  [Street-Filtered] Bỏ tên đường theo từ khóa: '%s'", text)
                #     continue

                valid_lines.append({
                    'text': text,
                    'left': line['left'],
                    'top': line['top'],
                    'width': line['width'],
                    'height': line['height'],
                    'conf': avg_conf,
                    'color': line_color
                })

        # 6. Nhận diện các ứng viên Icon trên bản đồ bằng OpenCV (multi-pass)
        scale = img_metadata.get("scale", 1.0) if img_metadata else 1.0
        candidate_icons = []
        if cv_img is not None:
            # Kích thước icon chuẩn ở zoom 19-21, nới để bắt icon rất nhỏ như UBND/landmark.
            min_icon_size = max(6, int(7 * scale))
            max_icon_size = int(46 * scale)

            def add_icon_candidate(cx_box: int, cy_box: int, cw_box: int, ch_box: int, source: str) -> None:
                if not (min_icon_size <= cw_box <= max_icon_size and min_icon_size <= ch_box <= max_icon_size):
                    return
                aspect_ratio = float(cw_box) / max(1, ch_box)
                if not (0.45 <= aspect_ratio <= 1.75):
                    return
                icon_bbox = [cx_box, cy_box, cx_box + cw_box, cy_box + ch_box]
                if not is_solid_icon(cv_img, icon_bbox):
                    return
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
                    'color': icon_color,
                    'source': source,
                })

            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 45, 145)
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                cx_box, cy_box, cw_box, ch_box = cv2.boundingRect(c)
                add_icon_candidate(cx_box, cy_box, cw_box, ch_box, "edge")

            hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
            color_mask = ((hsv[:, :, 1] > 45) & (hsv[:, :, 2] > 95)).astype(np.uint8) * 255
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                cx_box, cy_box, cw_box, ch_box = cv2.boundingRect(c)
                add_icon_candidate(cx_box, cy_box, cw_box, ch_box, "color")

            # Blob xám/tối nhỏ: bắt pin xám, icon landmark mảnh không đủ saturation.
            gray_mask = (((gray >= 55) & (gray <= 205))).astype(np.uint8) * 255
            gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
            contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < 12 * scale:
                    continue
                cx_box, cy_box, cw_box, ch_box = cv2.boundingRect(c)
                add_icon_candidate(cx_box, cy_box, cw_box, ch_box, "blob")

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

        # 6.5 Gộp lại các dòng OCR bị Tesseract tách ngang trong cùng một nhãn POI.
        # Trường hợp thực tế: icon Google Maps che/ép khoảng trắng làm tên "Ăn vặt - Nước Mía..."
        # bị tách thành "Ăn" và "vặt - Nước Mía...". Chỉ gộp khi có icon nằm sát bên trái
        # cùng hàng để tránh gộp nhầm các nhãn địa điểm độc lập phía trên bản đồ.
        def _merge_same_row_poi_label_fragments(lines_in: List[dict]) -> List[dict]:
            if not lines_in or not candidate_icons:
                return lines_in

            remaining = sorted(lines_in, key=lambda line: (line['top'], line['left']))
            merged_lines: List[dict] = []
            used = [False] * len(remaining)

            def has_left_icon_near(line_a: dict, line_b: dict) -> bool:
                top = min(line_a['top'], line_b['top'])
                bottom = max(line_a['top'] + line_a['height'], line_b['top'] + line_b['height'])
                center_y = (top + bottom) / 2.0
                left = min(line_a['left'], line_b['left'])
                right = max(line_a['left'] + line_a['width'], line_b['left'] + line_b['width'])
                for icon in candidate_icons:
                    same_row = abs(icon['y'] - center_y) <= max(18 * scale, (bottom - top) * 0.9)
                    left_adjacent = icon['right'] >= left - 55 * scale and icon['left'] <= left + 16 * scale
                    not_far_from_label = icon['left'] <= right + 20 * scale
                    if same_row and left_adjacent and not_far_from_label:
                        return True
                return False

            for i, base in enumerate(remaining):
                if used[i]:
                    continue
                current = dict(base)
                current['words'] = list(base.get('words', []))
                used[i] = True

                changed = True
                while changed:
                    changed = False
                    cur_right = current['left'] + current['width']
                    cur_bottom = current['top'] + current['height']
                    for j, other in enumerate(remaining):
                        if used[j]:
                            continue
                        other_right = other['left'] + other['width']
                        other_bottom = other['top'] + other['height']
                        overlap_y = min(cur_bottom, other_bottom) - max(current['top'], other['top'])
                        min_h = min(current['height'], other['height'])
                        gap_x = other['left'] - cur_right
                        same_band = overlap_y >= 0.35 * min_h or abs((current['top'] + cur_bottom) / 2.0 - (other['top'] + other_bottom) / 2.0) <= 12 * scale
                        close_gap = -8 * scale <= gap_x <= 55 * scale
                        same_color = _colors_are_similar(current.get('color'), other.get('color'), thresh_h=20, thresh_s=65, thresh_v=65)
                        if same_band and close_gap and same_color and has_left_icon_near(current, other):
                            current['text'] = f"{current['text']} {other['text']}".strip()
                            current['conf'] = (float(current.get('conf', 0)) + float(other.get('conf', 0))) / 2.0
                            current['left'] = min(current['left'], other['left'])
                            current['top'] = min(current['top'], other['top'])
                            new_right = max(cur_right, other_right)
                            new_bottom = max(cur_bottom, other_bottom)
                            current['width'] = new_right - current['left']
                            current['height'] = new_bottom - current['top']
                            current['color'] = current.get('color') if current.get('color') is not None else other.get('color')
                            used[j] = True
                            changed = True
                            logger.info("  [Line-Merged] Gộp mảnh tên POI cùng hàng: '%s'", current['text'])
                            break
                merged_lines.append(current)
            return merged_lines

        valid_lines = _merge_same_row_poi_label_fragments(valid_lines)

        # 7. Đối sánh các dòng chữ thô với các Icon lân cận (Many-to-One) trước khi chạy VietOCR
        #    Fix 2+5: Kết hợp khoảng cách + màu sắc icon để phân biệt đúng khi nhiều icon ứng viên
        line_matches = {} # l_idx -> (best_i_idx, dist)
        for l_idx, line in enumerate(valid_lines):
            lx = line['left']
            ly = line['top']
            lw = line['width']
            lh = line['height']
            line_center_x = lx + lw / 2.0
            line_center_y = ly + lh / 2.0
            line_color = line.get('color')
            
            candidates_for_line = []  # [(i_idx, dist, color_match_score)]
            
            for i_idx, icon in enumerate(candidate_icons):
                ix, iy = icon['x'], icon['y']
                
                horizontal_gap = DETECT_HORIZONTAL_GAP_MAX * scale
                vertical_gap = DETECT_VERTICAL_BELOW_GAP_MAX * scale
                
                is_left_or_right = (
                    (icon['right'] >= lx - horizontal_gap and icon['left'] <= lx + horizontal_gap)
                    or
                    (icon['left'] <= lx + lw + horizontal_gap and icon['right'] >= lx + lw - horizontal_gap)
                )
                is_same_row = abs(iy - line_center_y) <= max(18 * scale, lh * OCR_ICON_Y_ALIGN_RATIO)
                is_horizontal_adjacent = is_left_or_right and is_same_row

                text_below_icon = (
                    line_center_y >= icon['bottom']
                    and (line['top'] - icon['bottom']) <= vertical_gap
                    and max(0, min(line['left'] + line['width'], icon['right']) - max(line['left'], icon['left'])) >= min(lw, icon['w']) * 0.25
                )
                
                if is_horizontal_adjacent or text_below_icon:
                    # Tính khoảng cách dựa trên quan hệ bố cục gần nhất.
                    if is_horizontal_adjacent:
                        if ix < lx:
                            dist_x = lx - icon['right']
                        elif ix > lx + lw:
                            dist_x = icon['left'] - (lx + lw)
                        else:
                            dist_x = 0
                        dist_y = abs(iy - line_center_y)
                        layout_score = 0.0
                    else:
                        dist_x = abs(ix - line_center_x)
                        dist_y = max(0, line['top'] - icon['bottom'])
                        layout_score = 0.35  # text dưới icon là trường hợp hiếm, ưu tiên sau layout ngang
                    dist = (dist_x**2 + dist_y**2)**0.5
                    
                    # Tính điểm color match: icon cùng tông màu với text được ưu tiên.
                    # Google Maps cũng hay dùng icon màu nhưng tên POI màu đen/xám bên cạnh.
                    # Trường hợp này vẫn là nhãn địa điểm hợp lệ, chỉ xếp sau match cùng màu thật.
                    icon_color = icon.get('color')
                    color_match = _colors_are_similar(
                        line_color,
                        icon_color,
                        thresh_h=DETECT_COLOR_THRESH_H,
                        thresh_s=DETECT_COLOR_THRESH_S,
                        thresh_v=DETECT_COLOR_THRESH_V,
                    )
                    text_is_neutral = line_color is None or _is_low_saturation(line_color, thresh_s=50)
                    icon_is_colored = icon_color is not None and not _is_low_saturation(icon_color, thresh_s=50)
                    if color_match:
                        color_score = 0.0
                    elif text_is_neutral and icon_is_colored:
                        color_score = 0.15  # icon màu + chữ đen/xám cạnh bên vẫn là POI hợp lệ
                    else:
                        color_score = 1.0  # 0 = match tốt, 1 = mismatch rõ
                    
                    candidates_for_line.append((i_idx, dist, color_score + layout_score))
            
            if candidates_for_line:
                # Sắp xếp: ưu tiên color match trước, sau đó khoảng cách gần nhất
                # Nhưng chỉ ưu tiên màu khi khoảng cách chênh lệch < 30px (tránh chọn icon quá xa)
                candidates_for_line.sort(key=lambda c: (c[2], c[1]))
                best = candidates_for_line[0]
                
                # Nếu best theo color mà quá xa so với nearest, chọn nearest
                nearest = min(candidates_for_line, key=lambda c: c[1])
                if best[1] - nearest[1] > 30 * scale:
                    best = nearest
                
                i_idx, dist, _ = best
                if l_idx not in line_matches or dist < line_matches[l_idx][1]:
                    line_matches[l_idx] = (i_idx, dist)

        # 7.5 Đối sánh các dòng chưa khớp (unmatched lines) vào cùng icon với dòng đã khớp gần nó
        #      Fix 3: Kiểm tra cả màu ICON (không chỉ màu text) + siết threshold + mở rộng dist_y
        matched_line_indices = set(line_matches.keys())
        unmatched_line_indices = [idx for idx in range(len(valid_lines)) if idx not in matched_line_indices]
        for u_idx in unmatched_line_indices:
            u_line = valid_lines[u_idx]
            u_text = u_line.get('text', '').strip()
            u_tokens = [w for w in re.split(r'\W+', _strip_vietnamese_accents(u_text)) if w]
            if len(u_tokens) <= 1 and not _is_likely_place_name(u_text):
                logger.info("  [Fragment-Skipped] Bỏ mảnh OCR quá ngắn trước khi gom icon: '%s'", u_text)
                continue
            best_match_idx = None
            min_dist_y = 999999
            
            for m_idx in matched_line_indices:
                m_line = valid_lines[m_idx]
                
                # Khoảng cách dòng dọc gần nhau: nới để gom tên POI bị Tesseract tách thành nhiều dòng.
                dist_y = abs(u_line['top'] - m_line['top'])
                if dist_y > 52 * scale:
                    continue
                    
                # Căn lề trái/phải hoặc đè ngang; nới ngưỡng vì Google Maps label nhỏ dễ lệch vài chục px.
                left_aligned = abs(u_line['left'] - m_line['left']) <= 42 * scale
                right_aligned = abs((u_line['left'] + u_line['width']) - (m_line['left'] + m_line['width'])) <= 42 * scale
                horizontal_overlap = max(0, min(u_line['left'] + u_line['width'], m_line['left'] + m_line['width']) - max(u_line['left'], m_line['left'])) > 0
                near_same_text_block = abs((u_line['left'] + u_line['width'] / 2.0) - (m_line['left'] + m_line['width'] / 2.0)) <= 55 * scale
                
                if left_aligned or right_aligned or horizontal_overlap or near_same_text_block:
                    # Nới nhẹ so sánh màu text để gom đủ các mảnh tên cùng POI sau enhance.
                    text_colors_similar = _colors_are_similar(
                        u_line['color'], m_line['color'], 
                        thresh_h=20, thresh_s=65, thresh_v=65
                    )
                    
                    if not text_colors_similar:
                        continue
                    
                    # Fix 3 core: Kiểm tra thêm màu ICON đã ghép với matched line
                    # Nếu icon có màu sắc rõ ràng (không phải xám/đen), 
                    # unmatched text phải tương thích với icon color
                    m_icon_idx = line_matches[m_idx][0]
                    m_icon_color = candidate_icons[m_icon_idx].get('color')
                    
                    # Nếu icon có màu sắc rực rỡ (saturation cao), text unmatched cũng
                    # phải tương thích — tránh gộp text thuộc icon khác màu
                    if m_icon_color is not None and not _is_low_saturation(m_icon_color, thresh_s=50):
                        u_text_color = u_line.get('color')
                        # Text đen/xám (low sat) thì ok — nó thuộc bất kỳ icon nào cũng được
                        # Nhưng text có màu rõ ràng phải cùng tông với icon
                        if u_text_color is not None and not _is_low_saturation(u_text_color, thresh_s=50):
                            if not _colors_are_similar(u_text_color, m_icon_color, thresh_h=20, thresh_s=60, thresh_v=60):
                                continue
                    
                    if dist_y < min_dist_y:
                        min_dist_y = dist_y
                        best_match_idx = m_idx
                            
            if best_match_idx is not None:
                # Kế thừa icon của dòng đã khớp
                i_idx, _ = line_matches[best_match_idx]
                line_matches[u_idx] = (i_idx, min_dist_y)
                matched_line_indices.add(u_idx)

        def _text_tokens(text: str) -> List[str]:
            return [w for w in re.split(r'\W+', _strip_vietnamese_accents(text or "")) if w]

        def _has_brand_or_primary_signal(text: str) -> bool:
            text_clean = (text or "").strip()
            text_lower = text_clean.lower()
            tokens = _text_tokens(text_clean)
            if not tokens:
                return False
            context_keywords = {
                "ubnd", "uy", "uỷ", "ủy", "ban", "phuong", "phường", "quan", "quận",
                "vong", "vòng", "xoay", "tram", "trạm", "ben", "bến", "ga",
                "cong", "cổng", "tuong", "tượng", "dai", "đài", "nha", "nhà", "tho", "thờ",
                "chua", "chùa", "den", "đền", "bao", "bảo", "tang", "tàng", "buu", "bưu", "dien", "điện",
                "school", "hotel", "store", "coffee", "cafe", "tea", "mall", "plaza", "station",
            }
            if _is_likely_place_name(text_clean) or any(k in text_lower for k in context_keywords):
                return True
            uppercase_letters = sum(1 for c in text_clean if c.isalpha() and c.isupper())
            letters = sum(1 for c in text_clean if c.isalpha())
            has_brand_case = letters >= 4 and uppercase_letters / max(1, letters) >= 0.45
            has_many_specific_tokens = len(tokens) >= 3 and not _is_generic_name(text_clean.lower())
            return has_brand_case or has_many_specific_tokens

        def _line_color_compatible(a: dict, b: dict, icon: dict) -> bool:
            a_color = a.get('color')
            b_color = b.get('color')
            icon_color = icon.get('color')
            if _colors_are_similar(a_color, b_color, thresh_h=22, thresh_s=75, thresh_v=75):
                return True
            if _is_low_saturation(a_color, thresh_s=55) and _is_low_saturation(b_color, thresh_s=55):
                return True
            if icon_color is not None:
                if _is_low_saturation(a_color, thresh_s=55) or _colors_are_similar(a_color, icon_color, thresh_h=22, thresh_s=75, thresh_v=75):
                    if _is_low_saturation(b_color, thresh_s=55) or _colors_are_similar(b_color, icon_color, thresh_h=22, thresh_s=75, thresh_v=75):
                        return True
            return False

        def _line_near_other_icon(line: dict, current_icon_idx: int) -> bool:
            line_cx = line['left'] + line['width'] / 2.0
            line_cy = line['top'] + line['height'] / 2.0
            current_icon = candidate_icons[current_icon_idx]
            current_dist = ((line_cx - current_icon['x']) ** 2 + (line_cy - current_icon['y']) ** 2) ** 0.5
            for other_idx, other_icon in enumerate(candidate_icons):
                if other_idx == current_icon_idx:
                    continue
                other_dist = ((line_cx - other_icon['x']) ** 2 + (line_cy - other_icon['y']) ** 2) ** 0.5
                if other_dist + 12 * scale < current_dist:
                    return True
            return False

        def _should_expand_line_to_icon(seed_line: dict, candidate_line: dict, icon: dict, icon_idx: int) -> bool:
            text_raw = candidate_line.get('text', '').strip()
            text_tokens = [w for w in re.split(r'\W+', _strip_vietnamese_accents(text_raw)) if w]
            if len(text_raw) < 3 or not any(c.isalpha() for c in text_raw):
                return False
            if len(text_tokens) <= 1 and not _is_likely_place_name(text_raw):
                return False
            cand_bbox = [candidate_line['left'], candidate_line['top'], candidate_line['left'] + candidate_line['width'], candidate_line['top'] + candidate_line['height']]
            if _is_slanted_road_name(cv_img, cand_bbox) and not _is_likely_place_name(text_raw):
                return False
            if _line_near_other_icon(candidate_line, icon_idx):
                return False
            if not _line_color_compatible(seed_line, candidate_line, icon):
                return False

            seed_left, seed_right = seed_line['left'], seed_line['left'] + seed_line['width']
            cand_left, cand_right = candidate_line['left'], candidate_line['left'] + candidate_line['width']
            seed_cx = (seed_left + seed_right) / 2.0
            cand_cx = (cand_left + cand_right) / 2.0
            seed_cy = seed_line['top'] + seed_line['height'] / 2.0
            cand_cy = candidate_line['top'] + candidate_line['height'] / 2.0
            vertical_gap = max(0.0, max(seed_line['top'], candidate_line['top']) - min(seed_line['top'] + seed_line['height'], candidate_line['top'] + candidate_line['height']))
            vertical_close = vertical_gap <= DETECT_CONTEXT_VERTICAL_GAP_MAX * scale and abs(seed_cy - cand_cy) <= (DETECT_CONTEXT_VERTICAL_GAP_MAX + 28) * scale
            overlap_x = max(0.0, min(seed_right, cand_right) - max(seed_left, cand_left))
            min_w = max(1.0, min(seed_line['width'], candidate_line['width']))
            aligned = (
                abs(seed_left - cand_left) <= DETECT_CONTEXT_X_ALIGN_MAX * scale
                or abs(seed_right - cand_right) <= DETECT_CONTEXT_X_ALIGN_MAX * scale
                or abs(seed_cx - cand_cx) <= (DETECT_CONTEXT_X_ALIGN_MAX + 35) * scale
                or overlap_x >= 0.25 * min_w
            )
            same_row = abs(seed_cy - cand_cy) <= max(22 * scale, min(seed_line['height'], candidate_line['height']) * 1.4)
            horizontal_gap = max(cand_left, seed_left) - min(cand_right, seed_right)
            icon_between_or_adjacent = (
                icon['left'] <= max(seed_right, cand_right) + 14 * scale
                and icon['right'] >= min(seed_left, cand_left) - 14 * scale
                and abs(icon['y'] - cand_cy) <= max(28 * scale, candidate_line['height'] * 1.6)
            )
            geometry_ok = (vertical_close and aligned) or (same_row and horizontal_gap <= 90 * scale and icon_between_or_adjacent)
            if not geometry_ok:
                return False

            has_primary_signal = _has_brand_or_primary_signal(text_raw)
            description_but_contextual = _is_generic_name(text_raw.lower()) and not has_primary_signal
            if description_but_contextual:
                return False
            return True

        if DETECT_CONTEXT_EXPAND_ENABLED and candidate_icons:
            changed = True
            while changed:
                changed = False
                matched_snapshot = list(line_matches.items())
                unmatched_snapshot = [idx for idx in range(len(valid_lines)) if idx not in line_matches]
                for u_idx in unmatched_snapshot:
                    candidate_line = valid_lines[u_idx]
                    best = None
                    best_score = 999999.0
                    for m_idx, (icon_idx, _) in matched_snapshot:
                        seed_line = valid_lines[m_idx]
                        icon = candidate_icons[icon_idx]
                        if not _should_expand_line_to_icon(seed_line, candidate_line, icon, icon_idx):
                            continue
                        dy = abs((seed_line['top'] + seed_line['height'] / 2.0) - (candidate_line['top'] + candidate_line['height'] / 2.0))
                        dx = abs((seed_line['left'] + seed_line['width'] / 2.0) - (candidate_line['left'] + candidate_line['width'] / 2.0))
                        score = dy * 1.4 + dx * 0.35
                        if score < best_score:
                            best_score = score
                            best = (icon_idx, m_idx)
                    if best is not None:
                        icon_idx, seed_idx = best
                        line_matches[u_idx] = (icon_idx, best_score)
                        changed = True
                        logger.info(
                            "  [Context-Expanded] Gộp dòng tên gần POI: '%s' vào icon của '%s'",
                            candidate_line.get('text', ''), valid_lines[seed_idx].get('text', '')
                        )

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
            "century", "19th", "historical", "heritage", "museum", "monument", "shrine", "attraction", "tourist",
            "cathedral", "chapel", "basilica", "europe", "european", "style", "tyle", "gothic", "roman",
            "and", "or", "of", "in", "the", "a", "&", "to", "for", "with", "by", "-",
            "nhật", "bản", "hàn", "quốc", "pháp", "mỹ", "việt", "nam", "trung", "quốc", "thái", "lan",
            "vietnamese", "japanese", "korean", "french", "italian", "american", "thai", "western", "asian",
            "phong", "cách", "châu", "âu", "á", "thế", "kỷ", "đầu", "cuối", "từ", "xây", "dựng", "kiến", "trúc",
            "style", "europe", "european", "asia", "asian", "built", "construction", "architecture"
        }

        def is_description_line(text_str: str) -> bool:
            text_lower = text_str.lower().strip()
            
            # Blacklist cụ thể cho popup "Send Product Feedback" rác trên Google Maps
            _BLACKLIST_PHRASES = {
                "send product feedback",
                "tọa độ",
                "tọ độ", # typo phổ biến
                "tọa đo",
                "tọ đo"
            }
            
            for phrase in _BLACKLIST_PHRASES:
                if phrase in text_lower:
                    return True

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
        #    Fix 4: Validation màu sắc nhất quán trong cùng 1 nhóm icon
        pois = []
        for i_idx, l_indices in icon_to_lines.items():
            icon = candidate_icons[i_idx]
            icon_color = icon.get('color')
            
            # Sắp xếp các dòng theo thứ tự từ trên xuống dưới
            matched_lines_raw = [(idx, valid_lines[idx]) for idx in l_indices]
            matched_lines_raw.sort(key=lambda x: x[1]['top'])
            
            # Fix 4: Kiểm tra consistency — nếu icon có màu rõ ràng,
            # loại bỏ các dòng text có màu khác biệt rõ ràng so với icon
            # (dấu hiệu bị gộp nhầm từ POI khác)
            if icon_color is not None and not _is_low_saturation(icon_color, thresh_s=50):
                consistent_lines = []
                for (lidx, line) in matched_lines_raw:
                    line_clr = line.get('color')
                    line_text = line.get('text', '').strip()
                    line_center_y = line['top'] + line['height'] / 2.0
                    line_gap_x = min(abs(line['left'] - icon['right']), abs(icon['left'] - (line['left'] + line['width'])))
                    same_row_near_icon = (
                        abs(line_center_y - icon['y']) <= max(20 * scale, line['height'] * 1.2)
                        and line_gap_x <= 65 * scale
                    )
                    # Text đen/xám (low saturation) → luôn chấp nhận (chữ thường)
                    if line_clr is None or _is_low_saturation(line_clr, thresh_s=50):
                        consistent_lines.append((lidx, line))
                    # Text có màu rõ ràng → phải tương đồng với icon
                    elif _colors_are_similar(line_clr, icon_color, thresh_h=20, thresh_s=60, thresh_v=60):
                        consistent_lines.append((lidx, line))
                    # Ngoại lệ hẹp: Google Maps đôi khi vẽ text địa điểm/park bằng màu xanh chữ
                    # khác màu nền/icon. Nếu tên có keyword địa điểm và nằm cùng hàng sát icon thì giữ.
                    elif _is_likely_place_name(line_text) and same_row_near_icon:
                        consistent_lines.append((lidx, line))
                        logger.info(
                            "  [Color-Split-Keep] Giữ dòng địa điểm '%s' dù màu khác icon tại (%d,%d)",
                            line_text, int(icon['x']), int(icon['y'])
                        )
                    else:
                        logger.info(
                            "  [Color-Split] Loại bỏ dòng '%s' khỏi nhóm icon tại (%d,%d) do màu khác biệt",
                            line.get('text', '?'), int(icon['x']), int(icon['y'])
                        )
                matched_lines_raw = consistent_lines
            
            if not matched_lines_raw:
                continue
                
            matched_lines = [line for (_, line) in matched_lines_raw]
            
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
                # Truyền thêm icon_bbox để xóa icon chính xác trong vùng đọc
                text_read = _recognize_text_crop_vietocr(cv_img, line_bbox, fallback=line['text'], icon_bbox=icon.get('bbox'))
                text_cleaned = _clean_spelling(text_read)
                
                # Dòng đầu tiên thường là tên chính, không nên lọc bỏ trừ khi là blacklist rác
                is_first_line = (len(name_parts) == 0)
                
                if is_description_line(text_cleaned):
                    if is_first_line:
                        # Nếu là dòng duy nhất/đầu tiên nhưng chứa từ khóa địa điểm quan trọng (Nhà thờ, Tượng đài, UBND, brand...)
                        # thì vẫn giữ làm tên thay vì bỏ qua.
                        if _is_likely_place_name(text_cleaned) or _has_brand_or_primary_signal(text_cleaned):
                            name_parts.append(text_cleaned)
                        else:
                            logger.info("  [Desc-Filtered] Bỏ dòng đầu (không phải tên riêng): '%s'", text_cleaned)
                            continue
                    else:
                        if _has_brand_or_primary_signal(text_cleaned) and text_cleaned.lower() not in " ".join(name_parts).lower():
                            name_parts.append(text_cleaned)
                        else:
                            logger.info("  [Desc-Filtered] Bỏ dòng mô tả phụ: '%s'", text_cleaned)
                            continue
                else:
                    name_parts.append(text_cleaned)
                
            # Bỏ qua POI hoàn toàn nếu không có dòng tên hợp lệ
            if not name_parts:
                logger.info("  [POI-Filtered] Bỏ POI tại (%f, %f) vì không có tên hợp lệ sau khi lọc.", icon['x'], icon['y'])
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
            
            # Tính toán bbox bao phủ icon + toàn bộ các dòng được gộp.
            box_pad = DETECT_POI_BOX_PAD_PX * scale
            min_l = min([icon['left']] + [line['left'] for line in matched_lines]) - box_pad
            min_t = min([icon['top']] + [line['top'] for line in matched_lines]) - box_pad
            max_r = max([icon['right']] + [line['left'] + line['width'] for line in matched_lines]) + box_pad
            max_b = max([icon['bottom']] + [line['top'] + line['height'] for line in matched_lines]) + box_pad
            
            # [Smart-Filter] Loại bỏ rác ở góc phải dưới (Popup "Send Product Feedback" của Google Maps)
            # Dựa trên ảnh thực tế 2080x1240, popup thường ở x > 1800 và y > 1100
            if icon['x'] > 1800 and icon['y'] > 1100:
                logger.info("  [Spatial-Filtered] Bỏ rác ở góc phải dưới: '%s' tại (%d, %d)", full_name, icon['x'], icon['y'])
                continue

            # [Edge-Filter] Loại bỏ các mảnh vụn chữ bị cắt ở mép ảnh (padding area)
            # Dựa trên overlap=80px trong config, margin an toàn nhất là ~40-45px.
            # Các mảnh chữ như "êt", "ong" thường xuất hiện trong khoảng 0-50px từ mép.
            h_img_limit, w_img_limit = cv_img.shape[:2]
            edge_margin = 45 
            
            is_near_edge = (icon['x'] < edge_margin or icon['x'] > w_img_limit - edge_margin or 
                            icon['y'] < edge_margin or icon['y'] > h_img_limit - edge_margin)
            
            # Nếu ở gần biên mà tên quá ngắn (< 4 ký tự) hoặc chỉ có 1 từ ngắn, khả năng cao là mảnh vụn
            is_fragment = is_near_edge and (len(full_name) <= 3 or (len(full_name.split()) == 1 and len(full_name) < 5))
            
            if is_near_edge or is_fragment:
                reason = "mảnh chữ sát mép" if is_near_edge else "mảnh vụn OCR"
                logger.info("  [Edge-Filtered] Bỏ %s: '%s' tại (%d, %d)", reason, full_name, icon['x'], icon['y'])
                continue

            pois.append({
                "name": full_name,
                "x": icon['x'], # Tọa độ GPS tâm POI chính là tâm của biểu tượng (icon)
                "y": icon['y'],
                "confidence": avg_conf / 100.0,
                "bbox": [float(min_l), float(min_t), float(max_r - min_l), float(max_b - min_t)],
                "icon_bbox": [float(icon['left']), float(icon['top']), float(icon['right']), float(icon['bottom'])],
                "has_icon": True
            })

        # 9.5 Text-led fallback: giữ địa điểm có chữ trên ảnh dù icon bị bỏ sót/không có icon rõ.
        # Không chỉnh nội dung chữ; chỉ tạo candidate detect để tránh thiếu POI landmark/label nhỏ.
        matched_line_indices_final = set(line_matches.keys())
        h_img_limit, w_img_limit = cv_img.shape[:2]
        text_led_keywords = {
            "ubnd", "ủy ban", "uỷ ban", "phường", "quận", "vòng xoay", "công viên",
            "nhà thờ", "chùa", "đền", "bảo tàng", "bưu điện", "trạm", "bến", "ga",
            "trường", "bệnh viện", "khách sạn", "cafe", "coffee", "restaurant", "tea",
            "nhà hàng", "quán", "cửa hàng", "siêu thị", "plaza", "mall", "landmark",
        }

        def _is_text_led_poi_candidate(line: dict) -> bool:
            text_raw = line.get('text', '').strip()
            if len(text_raw) < 3 or not any(c.isalpha() for c in text_raw):
                return False
            text_lower = text_raw.lower()
            token_count = len([w for w in re.split(r'\W+', text_lower) if w])
            if token_count < 2 and not any(k in text_lower for k in text_led_keywords):
                return False
            if is_description_line(text_raw) and not any(k in text_lower for k in text_led_keywords):
                return False
            line_bbox = [line['left'], line['top'], line['left'] + line['width'], line['top'] + line['height']]
            if _is_slanted_road_name(cv_img, line_bbox) and not _is_likely_place_name(text_raw):
                return False
            edge_margin = 45
            cx = line['left'] + line['width'] / 2.0
            cy = line['top'] + line['height'] / 2.0
            if cx < edge_margin or cx > w_img_limit - edge_margin or cy < edge_margin or cy > h_img_limit - edge_margin:
                return False

            # Dòng POI trên Google Maps thường là nhiều từ, hoặc có keyword địa điểm.
            has_place_keyword = _is_likely_place_name(text_raw) or any(k in text_lower for k in text_led_keywords)
            enough_visual_text = line['width'] >= 32 * scale and line['height'] >= 6 * scale and token_count >= 2
            return has_place_keyword or enough_visual_text

        for l_idx, line in enumerate(valid_lines):
            if l_idx in matched_line_indices_final:
                continue
            if not _is_text_led_poi_candidate(line):
                continue

            line_cx = line['left'] + line['width'] / 2.0
            line_cy = line['top'] + line['height'] / 2.0
            near_existing_icon_group = False
            for matched_idx in matched_line_indices_final:
                matched_line = valid_lines[matched_idx]
                same_band = abs((matched_line['top'] + matched_line['height'] / 2.0) - line_cy) <= 24 * scale
                horizontal_touch = max(line['left'], matched_line['left']) <= min(line['left'] + line['width'], matched_line['left'] + matched_line['width']) + 42 * scale
                close_center = abs((matched_line['left'] + matched_line['width'] / 2.0) - line_cx) <= 95 * scale
                if same_band and (horizontal_touch or close_center):
                    near_existing_icon_group = True
                    break
            if near_existing_icon_group:
                logger.info("  [Text-Led-Skipped] Bỏ text fallback sát POI đã match icon: '%s'", line.get('text', ''))
                continue

            box_pad = DETECT_POI_BOX_PAD_PX * scale
            min_l = float(line['left'] - box_pad)
            min_t = float(line['top'] - box_pad)
            max_r = float(line['left'] + line['width'] + box_pad)
            max_b = float(line['top'] + line['height'] + box_pad)
            text_name = line.get('text', '').strip()
            pois.append({
                "name": text_name,
                "x": float(line['left'] + line['width'] / 2.0),
                "y": float(line['top'] + line['height'] / 2.0),
                "confidence": max(0.35, float(line.get('conf', 45.0)) / 100.0),
                "bbox": [min_l, min_t, float(max_r - min_l), float(max_b - min_t)],
                "has_icon": False
            })
            logger.info("  [Text-Led-POI] Thêm candidate địa điểm từ text chưa match icon: '%s'", text_name)

        global _last_candidate_icons, _last_valid_lines, _last_line_to_icon, _last_potential_line_matches
        _last_candidate_icons = candidate_icons
        _last_valid_lines = valid_lines
        _last_line_to_icon = line_matches
        _last_potential_line_matches = []

        def _merge_split_poi_boxes(raw_pois: List[dict]) -> List[dict]:
            """Gộp các POI bị tách nhưng bbox chồng lấn/cùng vùng icon-text."""
            def rect(p: dict):
                l, t, w, h = p.get("bbox", [0, 0, 0, 0])
                return float(l), float(t), float(l + w), float(t + h)

            def overlap_ratio(a: dict, b: dict) -> float:
                ax1, ay1, ax2, ay2 = rect(a)
                bx1, by1, bx2, by2 = rect(b)
                ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
                iy = max(0.0, min(ay2, by2) - max(ay1, by1))
                inter = ix * iy
                if inter <= 0:
                    return 0.0
                area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
                area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
                return inter / min(area_a, area_b)

            def should_merge(a: dict, b: dict) -> bool:
                ax1, ay1, ax2, ay2 = rect(a)
                bx1, by1, bx2, by2 = rect(b)
                cx_a, cy_a = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
                cx_b, cy_b = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                same_text_band = abs(cy_a - cy_b) <= 28 * scale
                close_x = abs(cx_a - cx_b) <= 95 * scale
                horizontal_gap = max(ax1, bx1) - min(ax2, bx2)
                same_row_adjacent = same_text_band and horizontal_gap <= 60 * scale

                # Hai candidate đều có icon thường là hai POI riêng. Chỉ merge nếu bbox đè mạnh
                # hoặc gần như cùng hàng; tránh gộp nhầm label trên/dưới như shop + bệnh viện.
                if a.get("has_icon") and b.get("has_icon"):
                    if abs(cy_a - cy_b) > 22 * scale and overlap_ratio(a, b) < 0.35:
                        return False

                # Text-led fallback có thể là phần đuôi tên bị OCR tách (vd: "Thuyền").
                # Chỉ gộp khi cùng hàng và sát ngang; không nới luật cho 2 icon thật.
                if a.get("has_icon") != b.get("has_icon") and same_row_adjacent:
                    icon_poi = a if a.get("has_icon") else b
                    fallback = b if not b.get("has_icon") else a
                    fallback_name = fallback.get("name", "").strip()
                    fallback_tokens = [w for w in re.split(r'\W+', fallback_name.lower()) if w]
                    fallback_is_short_fragment = len(fallback_tokens) <= 2 and len(fallback_name) <= 24
                    fallback_is_place_name = _is_likely_place_name(fallback_name)
                    fallback_is_description = is_description_line(fallback_name) and not fallback_is_place_name
                    icon_h = max(1.0, rect(icon_poi)[3] - rect(icon_poi)[1])
                    fallback_h = max(1.0, rect(fallback)[3] - rect(fallback)[1])
                    fallback_is_smaller_text = fallback_h <= icon_h * 0.72
                    if fallback_is_smaller_text and not fallback_is_place_name:
                        return False
                    if not fallback_is_description and (fallback_is_short_fragment or fallback_is_place_name):
                        return True

                return overlap_ratio(a, b) >= 0.18 or (same_text_band and close_x and max(ax1, bx1) <= min(ax2, bx2) + 18 * scale)

            merged: List[dict] = []
            for poi in raw_pois:
                target_idx = None
                for idx, existing in enumerate(merged):
                    if should_merge(existing, poi):
                        target_idx = idx
                        break
                if target_idx is None:
                    merged.append(poi)
                    continue

                existing = merged[target_idx]
                ex1, ey1, ex2, ey2 = rect(existing)
                px1, py1, px2, py2 = rect(poi)
                nx1, ny1 = min(ex1, px1), min(ey1, py1)
                nx2, ny2 = max(ex2, px2), max(ey2, py2)

                existing_name = existing.get("name", "").strip()
                poi_name = poi.get("name", "").strip()
                if poi_name and poi_name.lower() not in existing_name.lower():
                    if existing_name and existing_name.lower() not in poi_name.lower():
                        existing["name"] = f"{existing_name} {poi_name}".strip()
                    elif len(poi_name) > len(existing_name):
                        existing["name"] = poi_name

                existing["bbox"] = [float(nx1), float(ny1), float(nx2 - nx1), float(ny2 - ny1)]
                existing["confidence"] = max(float(existing.get("confidence", 0.0)), float(poi.get("confidence", 0.0)))
                if poi.get("has_icon") and not existing.get("has_icon"):
                    existing["x"] = poi.get("x")
                    existing["y"] = poi.get("y")
                    existing["has_icon"] = True
                else:
                    existing["x"] = existing.get("x") if existing.get("x") is not None else poi.get("x")
                    existing["y"] = existing.get("y") if existing.get("y") is not None else poi.get("y")
                logger.info("  [POI-Merged] Gộp POI bị tách: '%s' + '%s'", existing_name, poi_name)
            return merged

        pois = _merge_split_poi_boxes(pois)

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


def enhance_for_detection(image_bytes: bytes) -> bytes:
    """
    Enhance nhẹ ảnh gốc để detect icon/text tốt hơn mà không khử nền.
    Giữ màu và cấu trúc nền bản đồ; chỉ tăng tương phản cục bộ, sắc nét và độ rực vừa phải.
    """
    if not DETECT_ENHANCE_ENABLED:
        return image_bytes
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return image_bytes

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        l = clahe.apply(l)
        contrast_img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        blurred = cv2.GaussianBlur(contrast_img, (0, 0), 1.0)
        sharpened = cv2.addWeighted(contrast_img, 1.35, blurred, -0.35, 0)

        hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = np.clip(s.astype(np.float32) * 1.12, 0, 255).astype(np.uint8)
        v = np.clip(v.astype(np.float32) * 0.98, 0, 255).astype(np.uint8)
        enhanced = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

        success, encoded_img = cv2.imencode('.png', enhanced)
        if success:
            return encoded_img.tobytes()
    except Exception as e:
        logger.warning("Lỗi khi enhance ảnh detect: %s", e)
    return image_bytes


def remove_background(image_bytes: bytes) -> bytes:
    """
    Hàm này hiện tại không sử dụng vì hệ thống chuyển sang khử nền động trong từng box POI.
    Trả về ảnh gốc để đảm bảo không lỗi cú pháp.
    """
    return image_bytes


def save_poi_crop(image_bytes: bytes, poi: dict, output_path: str, crop_x: int = 0, crop_y: int = 0) -> bool:
    """
    Cắt vùng bbox của POI từ ảnh gốc và lưu thành file PNG (không xử lý).
    """
    try:
        import cv2
        import numpy as np
        import os

        # Decode BGR image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return False

        bbox = poi.get("bbox")
        if not bbox:
            return False

        left, top, w, h = bbox
        ix1, iy1 = int(left - crop_x), int(top - crop_y)
        ix2, iy2 = int(ix1 + w), int(iy1 + h)

        h_img, w_img = img.shape[:2]
        ix1 = max(0, min(ix1, w_img - 1))
        iy1 = max(0, min(iy1, h_img - 1))
        ix2 = max(0, min(ix2, w_img - 1))
        iy2 = max(0, min(iy2, h_img - 1))

        if ix2 <= ix1 or iy2 <= iy1:
            return False

        # Cắt và lưu ảnh màu nguyên bản
        crop = img[iy1:iy2, ix1:ix2]
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, crop)
        return True
    except Exception as e:
        logger.warning("Lỗi khi lưu crop POI gốc: %s", e)
        return False


def draw_detections(image_bytes: bytes, pois: List[dict], crop_x: int = 0, crop_y: int = 0, draw_text: bool = True) -> bytes:
    """
    Vẽ khung chữ nhật (bbox) và nhãn văn bản (name) của từng địa điểm đã nhận diện
    lên ảnh nền đã được khử. Hỗ trợ Unicode tiếng Việt bằng PIL.
    - image_bytes: bytes của ảnh nền đã khử (PNG/JPEG)
    - pois: danh sách các POI từ extract_pois_from_screenshot
    - crop_x, crop_y: offset cắt của ảnh lưu so với ảnh chụp full
    - draw_text: True nếu muốn vẽ cả nhãn chữ, False nếu chỉ vẽ khung đỏ
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

        if draw_text:
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


def save_poi_crop(image_bytes: bytes, poi: dict, output_path: str, crop_x: int = 0, crop_y: int = 0) -> bool:
    """
    Cắt vùng bbox của POI từ ảnh gốc (màu) và lưu trực tiếp.
    """
    try:
        import cv2
        import numpy as np
        import os

        # Decode BGR image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return False

        bbox = poi.get("bbox")
        if not bbox:
            return False

        left, top, w, h = bbox
        ix1, iy1 = int(left - crop_x), int(top - crop_y)
        ix2, iy2 = int(ix1 + w), int(iy1 + h)

        h_img, w_img = img.shape[:2]
        ix1 = max(0, min(ix1, w_img - 1))
        iy1 = max(0, min(iy1, h_img - 1))
        ix2 = max(0, min(ix2, w_img - 1))
        iy2 = max(0, min(iy2, h_img - 1))

        if ix2 <= ix1 or iy2 <= iy1:
            return False

        # Cắt ảnh gốc màu
        crop = img[iy1:iy2, ix1:ix2].copy()

        # Đảm bảo thư mục tồn tại
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        cv2.imwrite(output_path, crop)
        return True
    except Exception as e:
        logger.warning("Lỗi khi lưu crop POI: %s", e)
        return False


