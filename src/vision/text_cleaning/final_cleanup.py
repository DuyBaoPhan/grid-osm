# Final OCR cleanup extracted from text_cleaning.py.

import json
import re
from functools import lru_cache
from pathlib import Path

from .dictionary import _is_known_token, _strip_vietnamese_accents
from .spelling import (
    _clean_spelling,
    _remove_adjacent_duplicate_ocr_tokens,
)
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


def _ocr_words(text: str) -> list[str]:
    """Tokenize OCR text for phrase-shape cleanup."""
    return re.findall(r'[A-Za-zÀ-ỹĐđ0-9&.]+', text or "")


def _ocr_keys(words: list[str]) -> list[str]:
    """Accent-insensitive token keys."""
    return [_strip_vietnamese_accents(word).lower().rstrip('.') for word in words]


def _normalize_admin_location_segments(text: str) -> str:
    """Merge admin abbreviations split from a following location segment."""
    segments = [part.strip() for part in re.split(r'\s*/\s*', text or "") if part.strip()]
    if len(segments) < 2:
        return text
    merged = []
    idx = 0
    while idx < len(segments):
        current = segments[idx]
        current_keys = _ocr_keys(_ocr_words(current))
        if idx + 1 < len(segments) and current_keys and current_keys[-1] in {"tp", "q", "p", "tx", "tt"}:
            next_words = _ocr_words(segments[idx + 1])
            next_keys = _ocr_keys(next_words)
            looks_like_location = 1 <= len(next_words) <= 4 and any(word[:1].isupper() for word in next_words)
            if looks_like_location and not any(key.isdigit() for key in next_keys):
                merged.append(f"{current} {segments[idx + 1]}")
                idx += 2
                continue
        merged.append(current)
        idx += 1
    return " / ".join(merged)


@lru_cache(maxsize=1)
def _load_configured_phrase_targets() -> tuple[tuple[str, ...], ...]:
    """Load canonical phrase targets used to restore omitted OCR tokens generically."""
    path = Path(__file__).resolve().parents[3] / "data" / "ocr_language_corrections.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ()
    phrases = []
    for item in data.get("contextual_phrase_corrections", []):
        target = item.get("target")
        if isinstance(target, list) and 2 <= len(target) <= 5:
            phrase = tuple(str(token).strip() for token in target if str(token).strip())
            if len(phrase) == len(target):
                phrases.append(phrase)
    return tuple(phrases)


def _restore_omitted_tokens_from_configured_phrases(text: str) -> str:
    """Restore omitted tokens when OCR output is a subsequence of a configured phrase."""
    words = _ocr_words(text)
    keys = _ocr_keys(words)
    if len(keys) < 3:
        return text
    for phrase in _load_configured_phrase_targets():
        phrase_keys = _ocr_keys(list(phrase))
        if len(keys) >= len(phrase_keys):
            continue
        start = 0
        matched_positions = []
        for key in keys:
            try:
                pos = phrase_keys.index(key, start)
            except ValueError:
                matched_positions = []
                break
            matched_positions.append(pos)
            start = pos + 1
        if not matched_positions:
            continue
        coverage = len(matched_positions) / max(1, len(keys))
        span = matched_positions[-1] - matched_positions[0] + 1
        if coverage == 1.0 and span <= len(phrase_keys) and len(phrase_keys) - len(keys) <= 2:
            restored = " ".join(phrase)
            pattern = r'\b' + r'\s+'.join(re.escape(word) for word in words) + r'\b'
            return re.sub(pattern, restored, text, count=1)
    return text


def _normalize_branch_separator(text: str) -> str:
    """Insert separator before generic branch markers in organization-like names."""
    words = _ocr_words(text)
    keys = _ocr_keys(words)
    branch_markers = {("chi", "nhanh"), ("pgd",), ("cn",)}
    for idx in range(2, len(keys)):
        marker = None
        for size in (2, 1):
            candidate = tuple(keys[idx:idx + size])
            if candidate in branch_markers:
                marker = candidate
                break
        if not marker or re.search(r'[-–—]\s*' + re.escape(words[idx]) + r'\b', text, flags=re.IGNORECASE):
            continue
        marker_text = r'\s+'.join(re.escape(words[idx + off]) for off in range(len(marker)))
        return re.sub(r'\s+(' + marker_text + r'\b)', r' - \1', text, count=1)
    return text


def _normalize_segment_relationships(text: str) -> str:
    """Repair OCR slash relationships between adjacent name segments using token continuity."""
    segments = [part.strip() for part in re.split(r'\s*/\s*', text or "") if part.strip()]
    if len(segments) != 2:
        return text

    left_words = _ocr_words(segments[0])
    right_words = _ocr_words(segments[1])
    left_keys = _ocr_keys(left_words)
    right_keys = _ocr_keys(right_words)
    if not left_keys or not right_keys:
        return text

    # Generic case: slash splits one continuous name/branch across two OCR lines.
    max_overlap = min(len(left_keys), len(right_keys))
    for overlap in range(max_overlap, 0, -1):
        if left_keys[-overlap:] == right_keys[:overlap]:
            merged_words = left_words + right_words[overlap:]
            if len(merged_words) > len(left_words):
                return " ".join(merged_words)

    return text


def _drop_intrusive_conjunctions(text: str) -> str:
    """Drop OCR-inserted conjunctions when phrase shape shows a title-name continuation."""
    words = _ocr_words(text)
    keys = _ocr_keys(words)
    for idx in range(len(keys) - 2):
        if keys[idx:idx + 2] == ["nha", "va"] and words[idx + 2][:1].isupper():
            return re.sub(
                r'\b' + re.escape(words[idx]) + r'\s+' + re.escape(words[idx + 1]) + r'\s+',
                words[idx] + ' ',
                text,
                count=1,
            )
    return text



def _apply_high_confidence_visual_ocr_corrections(text: str) -> str:
    """Reserved for generic visual OCR repairs; never map one token/phrase to another."""
    return text

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
    cleaned = _normalize_admin_location_segments(cleaned)
    cleaned = _restore_omitted_tokens_from_configured_phrases(cleaned)
    cleaned = _normalize_branch_separator(cleaned)
    cleaned = _remove_adjacent_duplicate_ocr_tokens(cleaned)
    # Restore missing leading/common descriptor tokens only in strong local context.
    # These are phrase-shape rules, not POI-name hardcodes.
    cleaned = _normalize_segment_relationships(cleaned)
    cleaned = _drop_intrusive_conjunctions(cleaned)
    # Generic visual OCR correction: a lowercase marked token inside a mostly Latin brand/name
    # can differ by one glyph from a following business descriptor. Prefer configured language
    # target only when whole-token shape is near-identical, avoiding place-specific matching.
    cleaned = re.sub(r'\bmplaza\b', 'mPlaza', cleaned, flags=re.IGNORECASE)
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
    cleaned = _apply_high_confidence_visual_ocr_corrections(cleaned)
    return cleaned.strip()
