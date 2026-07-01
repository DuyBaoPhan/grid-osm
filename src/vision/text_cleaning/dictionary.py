# Dictionary/accent helpers extracted from text_cleaning.py.

import re

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


def _normalize_ocr_spelling_by_dictionary(text: str) -> str:
    """Sửa lỗi OCR theo dictionary/context ngoài vision.py, tránh hardcode POI cụ thể."""
    try:
        from src.vietnam_places import normalize_ocr_spelling
    except ImportError:
        try:
            from vietnam_places import normalize_ocr_spelling
        except ImportError:
            return text
    return normalize_ocr_spelling(text)


def _is_known_token(token: str) -> bool:
    """True nếu token là từ tiếng Việt / địa danh có trong từ điển."""
    try:
        from src.vietnam_places import is_known_token
    except ImportError:
        try:
            from vietnam_places import is_known_token
        except ImportError:
            return False
    return is_known_token(token)
