# OCR suffix/continuation rescue helpers extracted from text_cleaning.py.

import re
from typing import List

from .dictionary import _is_known_token, _strip_vietnamese_accents
from .spelling import _clean_spelling
from .junk import _is_junk_line, _looks_like_junk_token, _is_category_or_description_segment
from .quality import _junk_token_count, _looks_like_bad_ocr, _score_ocr_text_quality

# Imported lazily inside functions would avoid cycles, but function bodies are kept intact.
from .final_cleanup import _clean_final_ocr_text

def _continuation_tokens_are_valid(words: List[str]) -> bool:
    """True nếu phần nối thêm đủ giống tên địa điểm, không phải mô tả/rating rác."""
    if not words or len(words) > 5:
        return False
    continuation = " ".join(words)
    if _is_junk_line(continuation):
        return False
    has_vietnamese_or_title = any(
        re.search(r'[À-ỹĐđ]', word) or word[:1].isupper() or word.isupper()
        for word in words
    )
    if _looks_like_bad_ocr(continuation) and not has_vietnamese_or_title:
        return False
    if _junk_token_count(continuation) > 0:
        return False
    if re.search(r'\d+(?:[.,]\d+)?\s*(?:\(|★|\*)', continuation):
        return False
    digit_count = sum(ch.isdigit() for ch in continuation)
    letter_count = sum(ch.isalpha() for ch in continuation)
    if digit_count and digit_count / max(1, digit_count + letter_count) > 0.20:
        return False

    connector_keys = {"giao", "duong", "le", "street", "road", "corner", "nga", "tu", "xoay"}
    known_descriptor_keys = {
        "the", "coffee", "shop", "cafe", "tea", "house", "restaurant", "bar", "store",
        "quan", "ca", "phe", "tra", "sua", "nha", "hang", "tiem", "cho", "thach",
    }
    title_or_viet = 0
    connector_count = 0
    known_descriptor_count = 0
    for word in words:
        key = _strip_vietnamese_accents(word).lower()
        if key in connector_keys:
            connector_count += 1
            continue
        if word[:1].isupper() or word.isupper() or re.search(r'[À-ỹĐđ]', word):
            title_or_viet += 1
            continue
        if key in known_descriptor_keys or _is_known_token(word):
            known_descriptor_count += 1
            continue
        return False

    strong_count = title_or_viet + connector_count + known_descriptor_count
    if strong_count != len(words):
        return False
    if title_or_viet + connector_count >= 1:
        return True
    # All-lower category chains are usually Google category text, not missed name suffix.
    return len(words) <= 3 and any(_strip_vietnamese_accents(w).lower() in {"the", "nha", "tiem", "quan"} for w in words)


def _append_missing_known_suffix(base_text: str, alternate_text: str) -> str:
    """Cứu phần đuôi bị thiếu nếu alternate OCR chứa overlap + continuation sạch."""
    base = _clean_final_ocr_text(base_text)
    alt = _clean_final_ocr_text(_clean_spelling(alternate_text))
    raw_base = (base_text or "").strip()
    raw_base_words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', raw_base)
    base_display = base
    if raw_base_words:
        raw_base_keys = [_strip_vietnamese_accents(w).lower() for w in raw_base_words]
        cleaned_base_keys = [_strip_vietnamese_accents(w).lower() for w in re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', base)]
        if raw_base_keys == cleaned_base_keys:
            base_display = raw_base
    if not base or not alt or base == alt:
        return base

    base_words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', base)
    alt_words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', alt)
    raw_alt_words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', alternate_text or "")
    raw_alt_keys = [_strip_vietnamese_accents(w).lower() for w in raw_alt_words]
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
                continuation = " ".join(candidate_words).strip()
                if raw_alt_keys == alt_keys and next_idx + n <= len(raw_alt_words):
                    continuation = " ".join(raw_alt_words[next_idx:next_idx + n]).strip()
                if not continuation:
                    continue
                merged = f"{base} {continuation}"
                if _score_ocr_text_quality(merged) >= _score_ocr_text_quality(base) - 4:
                    return f"{base_display} {continuation}"

    return base


def _merge_missing_middle_tokens(base_text: str, alternate_text: str) -> str:
    """Merge alternate OCR when it contains base tokens plus a small clean missing span."""
    base = _clean_final_ocr_text(base_text)
    raw_alt = re.sub(r'\s+', ' ', (alternate_text or "").strip())
    if not base or not raw_alt:
        return base

    base_words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', base)
    alt_words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', raw_alt)
    if len(base_words) < 2 or len(alt_words) < len(base_words):
        return base

    base_keys = [_strip_vietnamese_accents(w).lower() for w in base_words]
    alt_keys = [_strip_vietnamese_accents(w).lower() for w in alt_words]

    if alt_keys == base_keys:
        # Same token sequence means no missing middle token exists.
        # Preserve selected/base OCR spelling instead of letting alternate overwrite diacritics.
        return base
    if len(alt_words) <= len(base_words):
        return base

    matched_alt_indexes = []
    search_from = 0
    for key in base_keys:
        try:
            idx = alt_keys.index(key, search_from)
        except ValueError:
            return base
        matched_alt_indexes.append(idx)
        search_from = idx + 1

    if not matched_alt_indexes:
        return base

    leading_extra = matched_alt_indexes[0]
    if leading_extra:
        # Extra leading text is usually neighboring label/icon noise, not missing middle text.
        return base

    extra_spans = []
    previous = -1
    for idx in matched_alt_indexes + [len(alt_words)]:
        if idx > previous + 1:
            extra_spans.append((previous + 1, idx))
        previous = idx

    if not extra_spans:
        return base
    if len(extra_spans) > 2:
        return base

    total_extra = sum(end - start for start, end in extra_spans)
    if total_extra < 1 or total_extra > 4:
        return base

    base_segment_count = len([p for p in re.split(r'\s*/\s*', base) if p.strip()])
    raw_alt_segments = [p.strip() for p in re.split(r'\s*/\s*', raw_alt) if p.strip()]
    if base_segment_count >= 2 and len(raw_alt_segments) > base_segment_count:
        alt_word_segment_indexes = []
        for seg_idx, segment in enumerate(raw_alt_segments):
            seg_words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', segment)
            alt_word_segment_indexes.extend([seg_idx] * len(seg_words))
        for start, end in extra_spans:
            if end == len(alt_words) and start < len(alt_word_segment_indexes):
                first_extra_segment = alt_word_segment_indexes[start]
                previous_segment = alt_word_segment_indexes[start - 1] if start > 0 else first_extra_segment
                if first_extra_segment > previous_segment:
                    return base

    for start, end in extra_spans:
        span_words = alt_words[start:end]
        span_text = " ".join(span_words)
        if not span_words:
            return base
        if _is_junk_line(span_text) or _junk_token_count(span_text) > 0:
            return base
        span_known = all(_is_known_token(w) for w in span_words)
        if _looks_like_bad_ocr(span_text) and not span_known and not any(re.search(r'[À-ỹĐđ]', w) or w[:1].isupper() or w.isupper() for w in span_words):
            return base
        if _is_category_or_description_segment(span_text):
            return base
        if re.search(r'\d+(?:[.,]\d+)?\s*(?:\(|★|\*)', span_text):
            return base

    if _score_ocr_text_quality(raw_alt) >= _score_ocr_text_quality(base) - 4:
        return _clean_final_ocr_text(raw_alt)
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


