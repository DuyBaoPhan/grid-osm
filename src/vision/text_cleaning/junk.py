# Junk/category filtering helpers extracted from text_cleaning.py.

import re

from .dictionary import _is_known_token, _strip_vietnamese_accents


def _ascii_key(text: str) -> str:
    return re.sub(r'[^a-z0-9]', '', _strip_vietnamese_accents(text or '').lower())


def _vowel_ratio(key: str) -> float:
    letters = [ch for ch in key if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(ch in 'aeiouy' for ch in letters) / len(letters)


def _looks_like_artifact_shape(token: str) -> bool:
    """Detect OCR artifact tokens by shape, not by exact junk word lists."""
    clean = re.sub(r'[^A-Za-zÀ-ỹĐđ0-9]', '', token or '')
    if not clean:
        return True
    if _is_known_token(clean):
        return False
    if re.search(r'[À-ỹĐđ]', clean):
        return False
    if clean.isupper() and 2 <= len(clean) <= 6:
        return False
    if clean[:1].isupper() and clean[1:].islower():
        key = _ascii_key(clean)
        if len(key) >= 4 and _vowel_ratio(key) >= 0.30:
            return False

    key = _ascii_key(clean)
    if not key:
        return True
    if any(ch.isdigit() for ch in key):
        return bool(re.search(r'[a-z]\d{2,}[a-z]', key))

    has_internal_caps = bool(re.search(r'[a-z][A-Z]{1,}', clean))
    repeated_fragments = len(re.findall(r'(?:con|com|col|cor|dis|trn|inter)', key)) >= 2
    long_low_vowel = len(key) >= 8 and _vowel_ratio(key) < 0.25
    edge_noise_tail = len(key) >= 7 and re.search(r'([a-z]{2})\1|(?:ss|cc|tt|nm|rn|mn|oum)$', key)
    improbable_prefix = len(key) >= 7 and re.match(r'^(?:dis|dist|contr|inter|trans|col|com|cor|trn)', key)
    short_prefix_noise = len(key) >= 6 and re.match(r'^(?:dir|dis|dist|col|com|con)', key) and re.search(r'([^aeiouy])\1', key)

    return bool(has_internal_caps or repeated_fragments or long_low_vowel or edge_noise_tail or improbable_prefix or short_prefix_noise)


def _is_junk_line(s: str) -> bool:
    """Loại bỏ các dòng chữ rác không phải là tên địa điểm."""
    s = (s or '').strip()
    if not s:
        return True

    if re.match(r'^\d+(\.\d+)?\s*\(\d+\)$', s) or re.match(r'^\(\d+\)$', s):
        return True
    if re.match(r'^\d+(\.\d+)?$', s):
        return True
    if re.search(r'^\d+(\.\d+)?%\s*\(\d+\)', s) or re.search(r'^\d+(\.\d+)?\s*\(\d+\)\s*-\s*', s):
        return True
    if re.search(r'\d{5,}', s) or re.match(r'^0{3,}', s):
        return True

    clean_s = re.sub(r'[^a-zA-Z0-9]', '', s)
    if not clean_s:
        return True

    digits = sum(c.isdigit() for c in clean_s)
    letters = sum(c.isalpha() for c in clean_s)
    total = len(clean_s)
    if total > 0 and digits >= 4 and (digits / total) > 0.65:
        if letters == 0 or (letters / total) < 0.20:
            return True

    words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9]+', s)
    if len(words) == 1:
        return _looks_like_artifact_shape(words[0])
    return False


def _clean_junk_words(s: str) -> str:
    kept = []
    for word in (s or '').split():
        clean = re.sub(r'[^A-Za-zÀ-ỹĐđ0-9]', '', word)
        if not clean:
            kept.append(word)
            continue
        if re.search(r'\d{4,}', clean) or re.match(r'^0{3,}', clean):
            continue
        if _looks_like_artifact_shape(clean):
            continue
        kept.append(word)

    res = ' '.join(kept).strip()
    return re.sub(r'^[\s,\-/]+|[\s,\-/]+$', '', res).strip()


def _looks_like_junk_token(token: str, *, is_edge: bool = False) -> bool:
    """Nhận diện token OCR vô nghĩa để không xuất ra kết quả cuối."""
    raw = (token or '').strip()
    clean = re.sub(r'[^A-Za-zÀ-ỹĐđ0-9]', '', raw)
    if not clean:
        return True
    if _is_known_token(clean):
        return False

    key = _ascii_key(clean)
    if key in {'tp', 'cn', 'k', 'q', 'p'}:
        return False
    if any(ch.isdigit() for ch in clean):
        return False
    if re.search(r'[À-ỹĐđ]', clean):
        return False
    if clean.isupper() and 2 <= len(clean) <= 6:
        return False
    if is_edge and len(clean) <= 1 and not clean[:1].isupper():
        return True
    return _looks_like_artifact_shape(clean)


def _drop_stray_leading_edge_token(part: str) -> str:
    """Drop a likely edge artifact when the remaining text is a strong POI name."""
    words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', part or '')
    if len(words) < 3:
        return part

    first = words[0]
    first_key = _ascii_key(first)
    if any(ch.isdigit() for ch in first) and not any(ch.isalpha() for ch in first):
        return part
    if re.search(r'[À-ỹĐđ]', first) and _is_known_token(first):
        return part
    if first.isupper() and len(first_key) > 1:
        return part

    rest = words[1:]
    rest_has_vietnamese = any(re.search(r'[À-ỹĐđ]', w) for w in rest)
    rest_title_or_upper = sum(1 for w in rest if w[:1].isupper() or w.isupper())
    strong_rest = rest_has_vietnamese and len(rest) >= 2 and rest_title_or_upper / max(1, len(rest)) >= 0.5

    if len(first_key) > 2:
        if strong_rest and not _is_known_token(first) and _looks_like_artifact_shape(first):
            return re.sub(r'^\s*' + re.escape(first) + r'\b\s*', '', part, count=1).strip()
        return part

    has_brand_signal = any(w.isupper() and len(w) >= 2 for w in rest) or rest_title_or_upper / max(1, len(rest)) >= 0.65
    if not has_brand_signal:
        return part
    if first_key in {'tp', 'q', 'p'}:
        return part
    if _is_known_token(first) and not re.search(r'[À-ỹĐđ]', first):
        return part
    return re.sub(r'^\s*' + re.escape(first) + r'\b\s*', '', part, count=1).strip()


def _is_category_or_description_segment(segment: str) -> bool:
    """Nhận diện segment mô tả/category Google Maps bằng hình dạng text, không dùng danh sách category."""
    words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', segment or '')
    if not words or len(words) > 5:
        return False
    has_digit = any(any(ch.isdigit() for ch in w) for w in words)
    if any(re.search(r'[À-ỹĐđ]', w) for w in words):
        return False
    if any(w.isupper() and 2 <= len(w) <= 6 for w in words):
        return False

    title_or_upper = sum(1 for w in words if w[:1].isupper() or w.isupper())
    known_count = sum(1 for w in words if _is_known_token(w))
    all_ascii_words = all(re.fullmatch(r'[A-Za-z0-9&]+', w) for w in words)
    mostly_lower = title_or_upper <= 1
    has_lower_descriptor_shape = any(re.search(r'[a-z]', w) for w in words)

    if has_digit:
        return len(words) >= 2 and all_ascii_words and title_or_upper <= 2 and has_lower_descriptor_shape
    return len(words) >= 2 and all_ascii_words and mostly_lower and known_count >= max(1, len(words) - 1)
