# OCR suffix/continuation rescue helpers extracted from text_cleaning.py.

import re
from typing import List

from .dictionary import _is_known_token, _strip_vietnamese_accents
from .spelling import _clean_spelling
from .junk import _is_junk_line, _looks_like_junk_token
from .quality import _junk_token_count, _looks_like_bad_ocr, _score_ocr_text_quality

# Imported lazily inside functions would avoid cycles, but function bodies are kept intact.
from .final_cleanup import _clean_final_ocr_text

def _continuation_tokens_are_valid(words: List[str]) -> bool:
    """True nếu phần nối thêm đủ giống tên địa điểm, không phải mô tả/rating rác."""
    if not words or len(words) > 5:
        return False
    continuation = " ".join(words)
    if _is_junk_line(continuation) or _looks_like_bad_ocr(continuation):
        return False
    if _junk_token_count(continuation) > 0:
        return False
    digit_count = sum(ch.isdigit() for ch in continuation)
    letter_count = sum(ch.isalpha() for ch in continuation)
    if digit_count and digit_count / max(1, digit_count + letter_count) > 0.20:
        return False

    connector_keys = {"giao", "duong", "le", "street", "road", "corner", "nga", "tu", "xoay"}
    title_or_viet = 0
    connector_count = 0
    for word in words:
        key = _strip_vietnamese_accents(word).lower()
        if key in connector_keys:
            connector_count += 1
            continue
        if word[:1].isupper() or re.search(r'[À-ỹĐđ]', word) or _is_known_token(word):
            title_or_viet += 1
    return title_or_viet >= 1 and (title_or_viet + connector_count) == len(words)


def _append_missing_known_suffix(base_text: str, alternate_text: str) -> str:
    """Cứu phần đuôi bị thiếu nếu alternate OCR chứa overlap + continuation sạch."""
    base = _clean_final_ocr_text(base_text)
    alt = _clean_final_ocr_text(_clean_spelling(alternate_text))
    if not base or not alt or base == alt:
        return base

    base_words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', base)
    alt_words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', alt)
    if len(base_words) < 2 or len(alt_words) <= len(base_words):
        return base

    base_keys = [_strip_vietnamese_accents(w).lower() for w in base_words]
    alt_keys = [_strip_vietnamese_accents(w).lower() for w in alt_words]
    max_tail = min(4, len(base_keys))

    for tail_len in range(max_tail, 0, -1):
        tail = base_keys[-tail_len:]
        for start in range(0, len(alt_keys) - tail_len):
            if alt_keys[start:start + tail_len] != tail:
                continue
            next_idx = start + tail_len
            continuation_words = alt_words[next_idx:next_idx + 5]
            if not continuation_words:
                continue

            # Chỉ cứu phần cùng segment hoặc phrase giao lộ/vị trí sạch.
            same_segment = False
            for seg in re.split(r'\s*/\s*', alt):
                seg_words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', seg)
                seg_keys = [_strip_vietnamese_accents(w).lower() for w in seg_words]
                for seg_start in range(0, len(seg_keys) - tail_len):
                    if seg_keys[seg_start:seg_start + tail_len] == tail and seg_start + tail_len < len(seg_keys):
                        same_segment = True
                        break
                if same_segment:
                    break
            if not same_segment:
                continue

            for n in range(min(5, len(continuation_words)), 0, -1):
                candidate_words = continuation_words[:n]
                if not _continuation_tokens_are_valid(candidate_words):
                    continue
                merged = f"{base} {' '.join(candidate_words)}"
                cleaned = _clean_final_ocr_text(merged)
                if _score_ocr_text_quality(cleaned) >= _score_ocr_text_quality(base) - 4:
                    return cleaned

    return base


def _merge_overlapping_ocr_continuation(base_text: str, alternate_text: str) -> str:
    """Merge OCR variants when one ends with tokens that start the other.

    Example shape: `A B C` + `C D E` -> `A B C D E`.
    Generic guard keeps only clean continuation segments with reasonable length.
    """
    base = _clean_final_ocr_text(base_text)
    alt = _clean_final_ocr_text(_clean_spelling(alternate_text))
    if not base or not alt or base == alt:
        return base

    base_words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', base)
    alt_words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', alt)
    if len(base_words) < 3 or len(alt_words) < 2:
        return base

    base_keys = [_strip_vietnamese_accents(w).lower() for w in base_words]
    alt_keys = [_strip_vietnamese_accents(w).lower() for w in alt_words]
    max_overlap = min(4, len(base_keys), len(alt_keys))

    for overlap in range(max_overlap, 0, -1):
        if base_keys[-overlap:] != alt_keys[:overlap]:
            continue
        continuation_words = alt_words[overlap:]
        if not continuation_words or len(continuation_words) > 5:
            return base
        continuation = " ".join(continuation_words)
        if _is_junk_line(continuation) or _junk_token_count(continuation) > 0:
            return base
        if _looks_like_bad_ocr(continuation):
            return base
        digit_count = sum(ch.isdigit() for ch in continuation)
        letter_count = sum(ch.isalpha() for ch in continuation)
        if digit_count and digit_count / max(1, digit_count + letter_count) > 0.25:
            return base
        merged = f"{base} {continuation}".strip()
        if _score_ocr_text_quality(merged) >= _score_ocr_text_quality(base) - 4:
            return _clean_final_ocr_text(merged)
        return base

    return base


