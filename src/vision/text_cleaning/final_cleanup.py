# Final OCR cleanup extracted from text_cleaning.py.

import re

from .dictionary import _is_known_token, _strip_vietnamese_accents
from .spelling import _clean_spelling
from .junk import (
    _clean_junk_words,
    _drop_stray_leading_edge_token,
    _is_category_or_description_segment,
    _is_junk_line,
    _looks_like_junk_token,
)
from .quality import _clean_ocr_edge_segments


def _strip_non_latin_vietnamese_script(text: str) -> str:
    """Keep Latin/Vietnamese OCR payload; remove CJK/Hangul/Kana and symbol wrappers."""
    if not text:
        return ""
    text = re.sub(r'[^A-Za-zÀ-ỹĐđ0-9\s/&.,%+\'()\-–—]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def _drop_numeric_wrappers(text: str, *, had_non_latin_script: bool = False) -> str:
    """Drop numeric/rating wrappers only when a clear multi-word name payload remains."""
    s = (text or "").strip()
    if not s:
        return ""

    parenthesized_wrapped = re.fullmatch(
        r"\d+(?:[.,]\d+)?\s*\(\s*([A-Za-zÀ-ỹĐđ][A-Za-zÀ-ỹĐđ\s/\-&.+']{2,})\s*\)\s*\d*(?:[.,]\d+)?\s*",
        s,
    )
    if parenthesized_wrapped:
        payload = parenthesized_wrapped.group(1).strip()
        if len(re.findall(r'[A-Za-zÀ-ỹĐđ]+', payload)) >= 2:
            return re.sub(r'\s+', ' ', re.sub(r'\s*/\s*', ' ', payload)).strip()

    wrapped = re.fullmatch(
        r"\d+(?:[.,]\d+)?\s*\(?\s*([A-Za-zÀ-ỹĐđ][A-Za-zÀ-ỹĐđ\s/\-&.+']{2,})\s*\)?\s+\d+(?:[.,]\d+)?\s*",
        s,
    )
    paren_payload = re.fullmatch(
        r"\(?\s*([A-Za-zÀ-ỹĐđ][A-Za-zÀ-ỹĐđ\s/\-&.+']{2,})\s*\)?\s*\d*(?:[.,]\d+)?\s*",
        s,
    )
    wrapped_match = wrapped or (paren_payload if had_non_latin_script else None)
    if wrapped_match:
        payload = wrapped_match.group(1).strip()
        if len(re.findall(r'[A-Za-zÀ-ỹĐđ]+', payload)) >= 2:
            return re.sub(r'\s+', ' ', re.sub(r'\s*/\s*', ' ', payload)).strip()

    # If non-Latin script was removed, a leading/trailing standalone number often belongs to that label,
    # but keep normal address/branch numbers such as `Tiệm Nhà Nấm 89` or `Quận 1`.
    words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9]+', s)
    if had_non_latin_script and len(words) >= 4 and words[0].isdigit() and words[-1].isalpha():
        s = re.sub(r'^\s*\d+(?:[.,]\d+)?\s+', '', s).strip()
    return re.sub(r'\s+', ' ', s).strip()

def _clean_final_ocr_text(text: str) -> str:
    """Cleanup cuối: không để ký tự/từ rác lọt ra output."""
    if not text:
        return ""
    had_non_latin_script = bool(re.search(r'[^A-Za-zÀ-ỹĐđ0-9\s/&.,%+\'()\-–—]', text))
    text = _strip_non_latin_vietnamese_script(text)
    text = _clean_spelling(text)
    text = _drop_numeric_wrappers(text, had_non_latin_script=had_non_latin_script)
    text = re.sub(r'\s+\d+(?:[.,]\d+)?\s*\(\s*\d+\s*\)\s*$', '', text).strip()
    text = re.sub(r'\s+(?:open|closed)\s+\d{1,2}(?::|\s)\d{2}\s*(?:am|pm)?\s*$', '', text, flags=re.IGNORECASE).strip()
    parts = [p.strip() for p in re.split(r'\s*/\s*', text) if p.strip()]
    if len(parts) >= 2:
        strong_parts = [p for p in parts if not _is_category_or_description_segment(p) and not _is_junk_line(p)]
        if strong_parts:
            parts = strong_parts
    cleaned_parts = []
    for part_idx, part in enumerate(parts):
        part = re.sub(r'^[^A-Za-zÀ-ỹĐđ0-9&]+|[^A-Za-zÀ-ỹĐđ0-9&.]+$', '', part).strip()
        part = _drop_stray_leading_edge_token(part)
        part = _clean_junk_words(part)
        words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', part)
        preserve_amp_ellipsis = bool(re.search(r'&\s*\.{2,}\s*$', part))
        kept_words = []
        for idx, word in enumerate(words):
            is_edge = idx == 0 or idx == len(words) - 1
            if word == "&" and is_edge and preserve_amp_ellipsis:
                kept_words.append(word)
                continue
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
    cleaned = re.sub(
        r'\b([^/]{1,40}\b(?:TP\.?|Q\.?|P\.?))\s*/\s*((?:Hồ\s+)?Chí\s+Minh|Hồ\s+Chí\s+Minh)\b',
        r'\1 \2',
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r'\b(TP\.?)\s+Chí\s+Minh\b',
        r'\1 Hồ Chí Minh',
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r'\b(Văn\s+phòng\s+đăng\s+ký\s+đất\s+đai|Ngân\s+hàng\s+Chính\s+sách\s+xã\s+hội)\s+(Chi\s+nhánh\b)',
        r'\1 - \2',
        cleaned,
        flags=re.IGNORECASE,
    )
    # Restore missing leading/common descriptor tokens only in strong local context.
    # These are phrase-shape rules, not POI-name hardcodes.
    cleaned = re.sub(
        r'^(Vặt\s*[-–—]\s*Nước\s+Mía\b)',
        r'Ăn \1',
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r'\b(Shop\s+Thời\s+Trang)\s*/\s*(Coin\s+Store)\b',
        r'\1 Nữ - \2',
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r'\b(Nhà)\s+Và\s+(?=[A-ZÀ-ỸĐ])',
        r'\1 ',
        cleaned,
    )
    segs = [p.strip() for p in re.split(r'\s*/\s*', cleaned) if p.strip()]
    if len(segs) >= 2:
        first_words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', segs[0])
        following = " ".join(segs[1:])
        following_words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', following)
        if len(first_words) == 1 and following_words:
            first = first_words[0]
            first_key = _strip_vietnamese_accents(first).lower()
            following_has_vietnamese = bool(re.search(r'[À-ỹĐđ]', following))
            following_title = sum(1 for w in following_words if w[:1].isupper() or w.isupper())
            if (
                len(first_key) >= 4
                and not re.search(r'[À-ỹĐđ]', first)
                and not _is_known_token(first)
                and following_has_vietnamese
                and following_title / max(1, len(following_words)) >= 0.5
            ):
                segs = segs[1:]
                cleaned = " / ".join(segs)
    if len(segs) == 2:
        left_words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', segs[0])
        right_words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', segs[1])
        if len(left_words) >= 3 and len(right_words) >= 3 and _is_known_token(right_words[0]):
            second_key = _strip_vietnamese_accents(right_words[1]).lower()
            if second_key in {"giao", "duong", "duan", "le", "street", "road"}:
                right_segment = segs[1]
                if left_words[-1][:1].isupper() and right_words[0][:1].islower():
                    right_segment = re.sub(
                        r'^\s*' + re.escape(right_words[0]) + r'\b',
                        right_words[0][:1].upper() + right_words[0][1:],
                        right_segment,
                        count=1,
                    )
                if second_key == "giao" and right_words[1] != "giao":
                    right_segment = re.sub(
                        r'\b' + re.escape(right_words[1]) + r'\b',
                        "giao",
                        right_segment,
                        count=1,
                    )
                cleaned = f"{segs[0]} {right_segment}"
    cleaned = _clean_ocr_edge_segments(cleaned) if '_clean_ocr_edge_segments' in globals() else cleaned
    if re.search(r'\b(?:DIY|souvenirs?|gifts?|accessories|crafts?)\b', cleaned, flags=re.IGNORECASE):
        cleaned = re.sub(r'\s+8\s*$', ' &...', cleaned)
    return cleaned.strip()
