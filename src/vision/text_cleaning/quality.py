# OCR quality/token/edge helpers extracted from text_cleaning.py.

import re
from typing import List

from .dictionary import _is_known_token, _strip_vietnamese_accents
from .spelling import _clean_spelling
from .junk import _is_category_or_description_segment, _is_junk_line, _looks_like_junk_token

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
    if len(words) == 1:
        token = words[0]
        key = _strip_vietnamese_accents(token).lower()
        if not _is_known_token(token) and not re.search(r'[À-ỹĐđ\d]', token):
            # Icon/edge hallucinations often become one weird Latin token:
            # mixed camel/allcaps tail (`ComonESSt`) or artifact prefixes (`Disted`).
            has_internal_upper = bool(re.search(r'[a-z][A-Z]{1,}', token))
            artifact_prefix = key.startswith((
                'com', 'con', 'col', 'cor', 'cos', 'cot', 'dis', 'dist',
                'dir', 'der', 'pr', 'trn', 'trans', 'inter', 'contra', 'hype'
            ))
            artifact_suffix = key.endswith((
                'esst', 'ess', 'sst', 'sted', 'chest', 'ness', 'cess',
                'tess', 'tracess', 'oum', 'natis', 'shone', 'ication',
                'ification', 'inten', 'inter', 'inters'
            ))
            if len(key) >= 6 and (has_internal_upper or (artifact_prefix and artifact_suffix)):
                return True
    if words and len(words) <= 2 and not any(ch.isdigit() for ch in s):
        vowel_count = sum(ch in 'aeiouyàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ' for ch in lower)
        letter_count = sum(ch.isalpha() for ch in lower)
        if letter_count and vowel_count / letter_count < 0.28:
            return True
    return False


def _looks_like_vietnamese_gibberish(text: str) -> bool:
    """Detect multi-token Vietnamese-looking OCR hallucination without hardcoding names."""
    words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', text or "")
    word_tokens = [w for w in words if w != "&"]
    if len(word_tokens) < 3:
        return False
    if any(any(ch.isdigit() for ch in w) for w in word_tokens):
        return False
    if any(w.isupper() and 2 <= len(w) <= 6 for w in word_tokens):
        return False
    if not any(re.search(r'[À-ỹĐđ]', w) for w in word_tokens):
        return False

    known_count = sum(1 for w in word_tokens if _is_known_token(w))

    unknown_marked = 0
    odd_repeated_sound = 0
    keys = []
    for w in word_tokens:
        key = _strip_vietnamese_accents(w).lower()
        keys.append(key)
        if re.search(r'[À-ỹĐđ]', w) and not _is_known_token(w):
            unknown_marked += 1
        if len(key) >= 3 and key[:1] in {'n', 't', 'v'} and key.endswith(('an', 'en', 'em', 'ien')):
            odd_repeated_sound += 1

    title_or_upper = sum(1 for w in word_tokens if w[:1].isupper() or w.isupper())
    first_unknown_marked = bool(re.search(r'[À-ỹĐđ]', word_tokens[0]) and not _is_known_token(word_tokens[0]))
    if known_count >= max(2, len(word_tokens) // 2) and unknown_marked < 2 and not (first_unknown_marked and title_or_upper <= 1):
        return False

    if first_unknown_marked and len(word_tokens) >= 4 and title_or_upper <= 1:
        return True

    if unknown_marked >= 2 and known_count == 0:
        return True
    if unknown_marked >= 2 and odd_repeated_sound >= 2:
        return True
    if len(set(keys)) <= len(keys) - 2 and unknown_marked >= 2:
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
    if _looks_like_vietnamese_gibberish(s):
        score -= 55

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
        leading_short_run = 0
        for word in seg_words[:4]:
            if (
                len(word) <= 3
                and not re.search(r'[À-ỹĐđ]', word)
                and not any(ch.isdigit() for ch in word)
                and (word[:1].isupper() or word.isupper())
            ):
                leading_short_run += 1
            else:
                break
        if leading_short_run >= 3 and len(seg_words) >= 5:
            score -= 120
        elif leading_short_run >= 2 and len(seg_words) >= 6:
            score -= 120
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
    score += len(re.findall(r"[^\w\sÀ-ỹ/&.,%+'\-–—]", s))
    for segment in re.split(r'\s*/\s*', s):
        words = re.findall(r'[A-Za-zÀ-ỹ0-9]+', segment)
        if len(words) > 1 and len(words[0]) == 1:
            score += 2
        if len(words) > 1 and len(words[-1]) == 1:
            score += 1
        for word in words:
            if any(ch.isdigit() for ch in word):
                continue
            viet_marks = len(re.findall(r'[À-ỹĐđ]', word))
            # Small-map OCR sometimes glues neighboring Vietnamese words into one
            # over-accented token, e.g. `Chínhồ`. Treat as artifact, not better text.
            if len(word) >= 6 and viet_marks >= 2 and word[:1].islower():
                score += 4
            if len(word) >= 7 and viet_marks >= 3:
                score += 5
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

    token_short_artifact_prefix = 0
    for tok in norm_tokens[:4]:
        if len(tok) <= 3 and re.fullmatch(r'[a-z]+', tok or ""):
            token_short_artifact_prefix += 1
        else:
            break
    if (
        len(primary_tokens) >= 4
        and token_short_artifact_prefix >= 2
        and ("/" in (norm_text or "") or len(norm_tokens) > len(primary_tokens))
    ):
        return True

    # Normalized cùng nội dung gần như Primary nhưng nhiều artifact hơn: giữ Primary.
    if overlap_ratio >= 0.70 and norm_artifacts > primary_artifacts:
        return True

    # Normalized không có artifact hơn, nhưng chỉ là biến thể chính tả/dấu câu của Primary.
    # Nếu Primary đã tốt, tránh override chỉ khi Normalized thực sự có thêm artifact.
    if overlap_ratio >= 0.78 and token_delta <= 2 and norm_artifacts > primary_artifacts:
        return True

    # Normalized thêm segment/từ ngoài khi primary đã đủ dài thường là ăn chữ nhãn cạnh crop.
    # CHỈ áp dụng khi có sự trùng lặp (overlap) hoặc primary là subsequence của normalized,
    # để tránh loại bỏ normalized khi primary là rác hoàn toàn khác biệt.
    if overlap_ratio >= 0.5 or _contains_token_subsequence(norm_tokens, primary_tokens):
        if len(primary_tokens) >= 4 and len(norm_tokens) > len(primary_tokens) + 1 and norm_artifacts >= primary_artifacts:
            return True

    short_artifact_tokens = sum(1 for tok in norm_tokens[:4] if len(tok) <= 3)
    if (
        len(primary_tokens) >= 4
        and overlap >= 2
        and short_artifact_tokens >= 3
        and len(norm_tokens) >= len(primary_tokens)
    ):
        return True

    admin_acronyms = {"ubnd", "hđnd", "hdnd", "tp", "hcm"}
    primary_admin = any(tok in admin_acronyms for tok in primary_tokens)
    norm_admin = any(tok in admin_acronyms for tok in norm_tokens)
    if primary_admin and not norm_admin and overlap_ratio < 0.75:
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


def _is_clean_short_brand_candidate(text: str) -> bool:
    """Primary 1 token brand/acronym sạch: không cho normalized unrelated thay thế."""
    words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', text or "")
    if len(words) != 1:
        return False
    token = words[0]
    if not (2 <= len(token) <= 6):
        return False
    if not token.isupper():
        return False
    if _junk_token_count(token) > 0:
        return False
    if _ocr_artifact_score(token) > 0:
        return False
    return True


def _texts_are_unrelated(left: str, right: str) -> bool:
    left_tokens = set(_ocr_tokens(left))
    right_tokens = set(_ocr_tokens(right))
    if not left_tokens or not right_tokens:
        return False
    return left_tokens.isdisjoint(right_tokens)


def _normalized_has_valid_main_name_extension(primary_text: str, norm_text: str) -> bool:
    """True nếu normalized = primary + phần tiếp theo có dạng tên chính, không phải mô tả/category."""
    primary_tokens = _ocr_tokens(_clean_spelling(primary_text))
    norm_tokens = _ocr_tokens(_clean_spelling(norm_text))
    if not primary_tokens or not norm_tokens or len(norm_tokens) <= len(primary_tokens):
        return False

    match_start = -1
    for start in range(0, len(norm_tokens) - len(primary_tokens) + 1):
        if norm_tokens[start:start + len(primary_tokens)] == primary_tokens:
            match_start = start
            break
    if match_start < 0:
        return False

    extra_leading = norm_tokens[:match_start]
    extra_trailing = norm_tokens[match_start + len(primary_tokens):]
    if extra_leading:
        # Extra phía trước dễ là chữ nhãn khác/icon hơn là tên bị cắt.
        return False
    if not extra_trailing:
        return False

    norm_parts = [p.strip() for p in re.split(r'\s*/\s*', norm_text or "") if p.strip()]
    primary_key = " ".join(primary_tokens)
    extra_parts = []
    seen_primary = False
    for part in norm_parts:
        part_tokens = _ocr_tokens(_clean_spelling(part))
        if not seen_primary and part_tokens and _contains_token_subsequence(part_tokens, primary_tokens):
            seen_primary = True
            continue
        if seen_primary:
            extra_parts.append(part)

    if not extra_parts:
        return False

    for part in extra_parts:
        tokens = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', part)
        word_tokens = [t for t in tokens if t != "&"]
        if not word_tokens:
            return False
        if _is_junk_line(part) or _looks_like_bad_ocr(part):
            return False

        meaningful_part = " ".join(word_tokens)
        if _junk_token_count(meaningful_part) > 0:
            return False
        if _is_category_or_description_segment(part):
            return False

        # Dòng mô tả/rating/category thường có nhiều dấu câu/số hoặc là câu dài viết thường.
        # Không dùng keyword riêng theo ngành/tỉnh để tránh hardcode theo trường hợp.
        if re.search(r'\d+(?:[.,]\d+)?\s*(?:\(|★|\*)', part):
            return False
        if re.search(r'\b\d{1,2}:\d{2}\b|\b\d{1,2}\s*(?:AM|PM|am|pm)\b', part):
            return False
        if re.search(r'\b\d+(?:st|nd|rd|th)\b', part, flags=re.IGNORECASE):
            return False
        digit_count = sum(ch.isdigit() for ch in part)
        letter_count = sum(ch.isalpha() for ch in part)
        if digit_count and digit_count / max(1, digit_count + letter_count) > 0.18:
            return False

        word_count = len(word_tokens)
        has_vietnamese = bool(re.search(r'[À-ỹĐđ]', part))
        title_or_upper = sum(1 for t in word_tokens if t[:1].isupper() or t.isupper())
        title_ratio = title_or_upper / max(1, word_count)
        has_acronym = any(t.isupper() and 2 <= len(t) <= 6 for t in word_tokens)
        has_name_separator = bool(re.search(r'[&+\-/]', part))
        mostly_lower = title_or_upper == 0
        descriptor_keys = {
            "tour", "group", "english", "chinese", "luxury", "online", "book",
            "near", "best", "city", "store", "shop", "restaurant", "coffee",
            "museum", "parking", "attraction", "fashion", "accessories",
        }
        part_keys = {_strip_vietnamese_accents(t).lower() for t in word_tokens}
        if not has_vietnamese and word_count >= 3 and (part_keys & descriptor_keys):
            return False

        if word_count > 7:
            return False
        if word_count >= 5 and mostly_lower:
            return False
        if word_count >= 4 and not (has_vietnamese or has_acronym or title_ratio >= 0.5 or has_name_separator):
            return False
        if not (has_vietnamese or has_acronym or title_ratio >= 0.5 or has_name_separator):
            return False

    return True


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
        shared_prefix = 0
        for p_tok, n_tok in zip(primary_tokens, norm_tokens):
            if p_tok != n_tok:
                break
            shared_prefix += 1
        if shared_prefix >= max(3, int(0.6 * min(len(primary_tokens), len(norm_tokens)))):
            remainder_text = " ".join(norm_tokens[shared_prefix:])
            norm_remainder = " ".join(re.findall(r'[A-Za-zÀ-ỹĐđ0-9]+', norm_text or "")[shared_prefix:])
            if (
                len(norm_tokens) > len(primary_tokens)
                or any(ch.isdigit() for ch in remainder_text)
                or _ocr_artifact_score(norm_remainder) > 0
                or _junk_token_count(norm_remainder) > 0
                or any(_is_category_or_description_segment(seg) for seg in re.split(r'\s*/\s*', norm_text or "")[1:])
            ):
                return True
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
    if re.fullmatch(r'&\s*\.{2,}', s):
        return False
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
