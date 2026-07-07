# Spelling/diacritic helpers extracted from text_cleaning.py.

import re

from .dictionary import (
    _is_known_token,
    _normalize_ocr_spelling_by_dictionary,
    _normalize_vietnamese_place_phrases,
    _strip_vietnamese_accents,
)
from .symspell_corrector import apply_symspell_ocr_corrections

def _choose_better_duplicate_token(left: str, right: str) -> str:
    """Chọn token tốt hơn khi 2 token OCR liền kề cùng base bỏ dấu."""
    if left == right:
        return left
    left_marks = len(re.findall(r'[À-ỹĐđ]', left or ""))
    right_marks = len(re.findall(r'[À-ỹĐđ]', right or ""))
    if right_marks > left_marks:
        return right
    if left_marks > right_marks:
        return left
    if right[:1].isupper() and not left[:1].isupper():
        return right
    return left


def _remove_adjacent_duplicate_ocr_tokens(text: str) -> str:
    """Xóa token OCR lặp liền kề bằng base bỏ dấu, không đụng brand ALLCAPS/số."""
    if not text:
        return text

    matches = list(re.finditer(r'[A-Za-zÀ-ỹĐđ0-9]+', text))
    if len(matches) < 2:
        return text

    replacements = {}
    remove_indexes = set()
    prev_key = None
    prev_token = None
    prev_idx = None
    for idx, match in enumerate(matches):
        token = match.group(0)
        key = _strip_vietnamese_accents(token).lower()
        if (
            prev_token is not None
            and prev_key == key
            and len(key) >= 2
            and not (
                prev_token != token
                and _has_vietnamese_mark(prev_token)
                and _has_vietnamese_mark(token)
            )
            and (
                not (prev_token.isupper() and token.isupper() and len(key) >= 2)
                or (prev_token == token and _has_vietnamese_mark(token))
                or key in {"tp", "q", "p", "tx", "tt", "cn", "ubnd", "hcm"}
            )
            and not any(ch.isdigit() for ch in prev_token + token)
            and not ((len(prev_token) == 1 or len(token) == 1) and key not in {"q", "p"})
        ):
            keep = _choose_better_duplicate_token(prev_token, token)
            replacements[prev_idx] = keep
            remove_indexes.add(idx)
            prev_token = keep
            continue
        prev_key = key
        prev_token = token
        prev_idx = idx

    if not remove_indexes and not replacements:
        return text

    out = []
    last = 0
    for idx, match in enumerate(matches):
        if idx in remove_indexes:
            out.append(text[last:match.start()].rstrip())
            last = match.end()
            continue
        out.append(text[last:match.start()])
        out.append(replacements.get(idx, match.group(0)))
        last = match.end()
    out.append(text[last:])
    return re.sub(r'\s+', ' ', ''.join(out)).strip()


def _latin_edit_distance(a: str, b: str) -> int:
    """Small Levenshtein distance for neighboring OCR brand-token cleanup."""
    a = (a or "").lower()
    b = (b or "").lower()
    if abs(len(a) - len(b)) > 2:
        return 3
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _fix_repeated_noisy_brand_prefix(text: str) -> str:
    """Drop OCR duplicate/noisy leading brand token when next token is same shape."""
    words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', text or "")
    if len(words) < 3:
        return text
    first, second = words[0], words[1]
    if not (first.isupper() and second.isupper() and 3 <= len(first) <= 6 and 3 <= len(second) <= 6):
        return text
    if _latin_edit_distance(first, second) > 1:
        return text
    rest = words[2:]
    if not rest or not any(w[:1].isupper() or re.search(r'[À-ỹĐđ]', w) for w in rest):
        return text
    keep = first if (_is_known_token(first) or first == second) else second
    return re.sub(r'^\s*' + re.escape(first) + r'\s+' + re.escape(second) + r'\b', keep, text, count=1)


def _fix_latin_brand_ocr_artifacts(text: str) -> str:
    """Fix generic OCR artifacts in Latin/brand tokens without changing Vietnamese words."""
    if not text:
        return text
    text = _fix_repeated_noisy_brand_prefix(text)

    # Vietnamese hyphen artifact between word tokens: `Tòa-nhà` -> `Tòa nhà`.
    text = re.sub(r'(?<=[A-Za-zÀ-ỹĐđ])[-–—](?=[A-Za-zÀ-ỹĐđ])', ' ', text)

    def _fix_crossed_d(match):
        token = match.group(0)
        # OCR sometimes reads Latin capital D as Vietnamese Đ in English brand words.
        # Only change when rest is plain ASCII and token has no other Vietnamese mark.
        if re.fullmatch(r'Đ[A-Za-z]{3,}', token):
            return 'D' + token[1:]
        return token

    text = re.sub(r'\bĐ[A-Za-z]{3,}\b', _fix_crossed_d, text)

    def _fix_stray_leading_capital(match):
        token = match.group(0)
        # OCR/icon edge can glue one uppercase letter before a normal TitleCase token:
        # `LPost`, `LEfora`. Do not touch real camel/acronym brands like `LPBank`.
        if not re.fullmatch(r'[A-Z][A-Z][a-z]{3,}', token):
            return token
        if len(token) >= 3 and token[2].isupper():
            return token
        candidate = token[1:]
        if re.fullmatch(r'[A-Z][a-z]*ola', candidate):
            candidate = candidate[:-3] + 'ora'
        key = _strip_vietnamese_accents(candidate).lower()
        vowels = sum(ch in 'aeiouy' for ch in key)
        letters = sum(ch.isalpha() for ch in key)
        if letters >= 4 and vowels / max(1, letters) >= 0.25:
            return candidate
        return token

    text = re.sub(r'\b[A-Z][A-Z][a-z]{3,}\b', _fix_stray_leading_capital, text)

    english_context_keys = {
        "central", "post", "office", "store", "plaza", "mall", "center", "centre",
        "coffee", "cafe", "restaurant", "hotel", "shop", "market", "bank", "branch",
        "corner", "lounge", "studio", "city", "saigon", "hcmc", "vietnam",
    }

    def _strip_accents_from_noisy_allcaps_brand(match):
        token = match.group(0)
        key = _strip_vietnamese_accents(token)
        key_upper = key.upper()
        # OCR can add Vietnamese marks to short Latin/brand acronyms: `TÚMI` -> `TUMI`.
        # Only do this in English/brand context and never for known Vietnamese tokens.
        if (
            2 <= len(token) <= 6
            and token.isupper()
            and re.search(r'[À-ỹ]', token)
            and re.fullmatch(r'[A-Z0-9&]+', key_upper)
            and not _is_known_token(token)
            and not _is_known_token(key)
        ):
            return key_upper
        return token

    words = re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', text or "")
    keys = [_strip_vietnamese_accents(word).lower() for word in words]
    has_english_brand_context = any(key in english_context_keys for key in keys)
    if has_english_brand_context:
        text = re.sub(r'\b[A-ZÀ-ỹĐ0-9&]{2,6}\b', _strip_accents_from_noisy_allcaps_brand, text)

    return text


def _clean_spelling(text: str) -> str:
    # Loại bỏ quote/bracket/backslash rác từ icon/viền crop.
    # Giữ apostrophe nằm giữa chữ Latin cho brand hợp lệ như Italiani's.
    text = re.sub(r"[\"`\\\[\]\{\}]", "", text)
    text = re.sub(r"(?<![A-Za-zÀ-ỹĐđ])'|'(?![A-Za-zÀ-ỹĐđ])", "", text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = _fix_latin_brand_ocr_artifacts(text)
    text = _normalize_vietnamese_place_phrases(text)
    text = _normalize_ocr_spelling_by_dictionary(text)
    text = apply_symspell_ocr_corrections(text)
    text = _remove_adjacent_duplicate_ocr_tokens(text)
    return text


def _has_vietnamese_mark(token: str) -> bool:
    """True nếu token có dấu tiếng Việt, gồm cả chữ đ/Đ."""
    return bool(re.search(r'[À-ỹĐđ]', token or ""))


def _merge_primary_diacritics(primary_text: str, selected_text: str) -> str:
    """
    Khi selected/normalized đúng cấu trúc hơn nhưng sai dấu nhẹ, mượn dấu từ Primary.
    Chỉ thay token nếu:
    - token bỏ dấu giống nhau
    - cả Primary và selected đều có dấu Việt
    - không đổi số/ký tự brand không dấu
    Ví dụ: Thợ -> Thọ sẽ khôi phục Thợ; Xe không bị đổi thành Xẻ.
    """
    primary_tokens = re.findall(r'[A-Za-zÀ-ỹĐđ0-9]+', primary_text or "")
    selected_tokens = re.findall(r'[A-Za-zÀ-ỹĐđ0-9]+', selected_text or "")
    if not primary_tokens or len(primary_tokens) != len(selected_tokens):
        return selected_text

    replacements = []
    changed = False
    for p_tok, s_tok in zip(primary_tokens, selected_tokens):
        if p_tok == s_tok:
            replacements.append(s_tok)
            continue
        if (
            _strip_vietnamese_accents(p_tok) == _strip_vietnamese_accents(s_tok)
            and _has_vietnamese_mark(p_tok)
            and _has_vietnamese_mark(s_tok)
            and not any(ch.isdigit() for ch in p_tok + s_tok)
        ):
            replacements.append(p_tok)
            changed = True
        else:
            replacements.append(s_tok)

    if not changed:
        return selected_text

    repl_iter = iter(replacements)
    return re.sub(r'[A-Za-zÀ-ỹĐđ0-9]+', lambda _: next(repl_iter), selected_text)


def _vietnamese_tone_preference(token: str) -> int:
    """Heuristic nhẹ để chọn dấu Việt tự nhiên hơn giữa hai token cùng base."""
    t = token or ""
    score = 0
    # Dấu sắc/huyền/nặng thường ổn định hơn hỏi/ngã trong OCR nhỏ; không ép tuyệt đối.
    score += 2 * len(re.findall(r'[áéíóúýấắếốớứ]', t, flags=re.IGNORECASE))
    score += 1 * len(re.findall(r'[àèìòùỳầằềồờừ]', t, flags=re.IGNORECASE))
    score += 1 * len(re.findall(r'[ạẹịọụỵậặệộợự]', t, flags=re.IGNORECASE))
    score -= 1 * len(re.findall(r'[ảẻỉỏủỷẩẳểổởử]', t, flags=re.IGNORECASE))
    score -= 1 * len(re.findall(r'[ãẽĩõũỹẫẵễỗỡữ]', t, flags=re.IGNORECASE))
    return score


def _has_vietnamese_shaped_vowel(token: str) -> bool:
    """True nếu token có nguyên âm Việt đặc thù dễ bị OCR làm mất: ơ/ư/ă/â/ê/ô."""
    return bool(re.search(r'[ăằắẳẵặâầấẩẫậêềếểễệôồốổỗộơờớởỡợưừứửữự]', token or "", flags=re.IGNORECASE))


def _merge_best_diacritics(primary_text: str, selected_text: str) -> str:
    """Chọn dấu tốt hơn giữa Primary và selected khi token cùng base bỏ dấu."""
    primary_tokens = re.findall(r'[A-Za-zÀ-ỹĐđ0-9]+', primary_text or "")
    selected_tokens = re.findall(r'[A-Za-zÀ-ỹĐđ0-9]+', selected_text or "")
    if not primary_tokens or len(primary_tokens) != len(selected_tokens):
        return selected_text

    replacements = []
    changed = False
    for p_tok, s_tok in zip(primary_tokens, selected_tokens):
        if p_tok == s_tok:
            replacements.append(s_tok)
            continue
        if (
            _strip_vietnamese_accents(p_tok) == _strip_vietnamese_accents(s_tok)
            and _has_vietnamese_mark(p_tok)
            and _has_vietnamese_mark(s_tok)
            and not any(ch.isdigit() for ch in p_tok + s_tok)
        ):
            # If selected token is plain ASCII but primary only adds Vietnamese tone,
            # do not reintroduce accent for brand/English-like names.
            if re.fullmatch(r'[A-Za-z]+', s_tok) and not _has_vietnamese_shaped_vowel(p_tok):
                replacements.append(s_tok)
            elif _has_vietnamese_shaped_vowel(p_tok) and not _has_vietnamese_shaped_vowel(s_tok):
                replacements.append(p_tok)
                changed = True
            elif _vietnamese_tone_preference(p_tok) > _vietnamese_tone_preference(s_tok):
                replacements.append(p_tok)
                changed = True
            else:
                replacements.append(s_tok)
        else:
            replacements.append(s_tok)

    if not changed:
        return selected_text
    repl_iter = iter(replacements)
    return re.sub(r'[A-Za-zÀ-ỹĐđ0-9]+', lambda _: next(repl_iter), selected_text)
