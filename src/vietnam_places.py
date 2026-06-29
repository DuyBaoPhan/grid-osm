"""Vietnam place-name normalization for OCR post-processing.

This module keeps the place gazetteer outside vision.py and applies conservative
phrase-level correction only. It is designed to fix OCR accent/noise errors in
Vietnamese place names without rewriting brand names.

Extension: OSM-based word dictionary (data/osm_words.json) provides single-token
and short-phrase corrections derived from OpenStreetMap name data for the
Ho Chi Minh City area. This supplements the phrase gazetteer with common
Vietnamese service/location words that OCR frequently drops accents on.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    from rapidfuzz import fuzz, process
except Exception:  # pragma: no cover - optional runtime dependency fallback
    fuzz = None
    process = None


_TOKEN_RE = re.compile(r"[A-Za-zÀ-ỹĐđ0-9]+")
_MAX_NGRAM = 6
_MIN_FUZZY_SCORE = 97

# Maximum n-gram size to look up in OSM word dictionary
_OSM_MAX_NGRAM = 4


_ACCENT_MAP = (
    (r"[àáảãạăằắẳẵặâầấẩẫậ]", "a"),
    (r"[èéẻẽẹêềếểễệ]", "e"),
    (r"[ìíỉĩị]", "i"),
    (r"[òóỏõọôồốổỗộơờớởỡợ]", "o"),
    (r"[ùúủũụưừứửữự]", "u"),
    (r"[ỳýỷỹỵ]", "y"),
    (r"[đ]", "d"),
)

# Ký tự tiếng Việt đặc trưng (nguyên âm mở rộng, phụ âm đ)
_VIET_SHAPED = re.compile(
    r"[ăằắẳẵặâầấẩẫậêềếểễệôồốổỗộơờớởỡợưừứửữựđĂẶÂẬÊỆÔỘƠỢƯỰĐ]"
)
# Bất kỳ dấu tiếng Việt nào (gồm cả dấu thanh)
_VIET_ANY = re.compile(
    r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    r"ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ]"
)


def strip_vietnamese_accents(text: str) -> str:
    """Strip Vietnamese accents and lowercase for stable matching."""
    s = (text or "").lower()
    for pattern, repl in _ACCENT_MAP:
        s = re.sub(pattern, repl, s)
    return s


def _has_vietnamese_mark(token: str) -> bool:
    """True nếu token có dấu tiếng Việt bất kỳ."""
    return bool(_VIET_ANY.search(token or ""))


def _has_shaped_vowel(token: str) -> bool:
    """True nếu token có nguyên âm tiếng Việt đặc trưng (ă/â/ê/ô/ơ/ư/đ)."""
    return bool(_VIET_SHAPED.search(token or ""))


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _load_places() -> Tuple[Dict[Tuple[str, ...], Tuple[str, ...]], List[str], Dict[str, Tuple[str, ...]]]:
    """Load gazetteer as exact token map and normalized phrase lookup."""
    path = _project_root() / "data" / "vietnam_places.txt"
    exact: Dict[Tuple[str, ...], Tuple[str, ...]] = {}
    fuzzy_choices: List[str] = []
    fuzzy_lookup: Dict[str, Tuple[str, ...]] = {}

    if not path.exists():
        return exact, fuzzy_choices, fuzzy_lookup

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        phrase = raw_line.strip()
        if not phrase or phrase.startswith("#"):
            continue
        canonical_tokens = tuple(m.group(0) for m in _TOKEN_RE.finditer(phrase))
        if not canonical_tokens:
            continue
        key = tuple(strip_vietnamese_accents(tok) for tok in canonical_tokens)
        exact[key] = canonical_tokens
        norm_phrase = " ".join(key)
        fuzzy_choices.append(norm_phrase)
        fuzzy_lookup[norm_phrase] = canonical_tokens

    # Longest phrases first helps deterministic exact matching.
    fuzzy_choices.sort(key=lambda item: (-len(item.split()), item))
    return exact, fuzzy_choices, fuzzy_lookup


@lru_cache(maxsize=1)
def _load_osm_words() -> Dict[str, str]:
    """
    Load OSM-derived word dictionary từ data/osm_words.json.
    Format: {"base_không_dấu": "Canonical_có_dấu"}
    
    Dictionary này được xây dựng từ tên POI/đường phố trên OSM trong khu vực TP.HCM.
    Mục đích: sửa dấu OCR cho các từ/cụm phổ biến trong tên địa điểm.
    KHÔNG dùng để map đến địa điểm cụ thể.
    """
    path = _project_root() / "data" / "osm_words.json"
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _is_brand_like(tokens: Tuple[str, ...]) -> bool:
    """Avoid rewriting likely brands/acronyms."""
    if not tokens:
        return True
    if any(any(ch.isdigit() for ch in tok) for tok in tokens):
        return True
    if any(tok.isupper() and len(tok) > 1 for tok in tokens):
        # Allow administrative abbreviation TP in place phrases.
        return any(tok not in {"TP"} for tok in tokens)
    return False


def _is_token_brand_like(token: str) -> bool:
    """
    True nếu token đơn này trông như brand/acronym không nên sửa dấu.
    Bảo thủ hơn _is_brand_like để tránh sửa nhầm tên riêng.
    """
    if not token:
        return True
    # Số hoặc số+chữ
    if any(ch.isdigit() for ch in token):
        return True
    # ALLCAPS 2+ ký tự → likely acronym/brand (AO, DAI, MCM, LPBank w/ leading caps...)
    if token.isupper() and len(token) >= 2:
        # Allow very common Vietnamese particles written uppercase
        # e.g. 'TP' is an administrative abbreviation we want to keep
        allowed_allcaps = {'TP'}
        if token not in allowed_allcaps:
            return True
    # Tiếng Anh thuần (không có ký tự Việt, không có thêm ký hiệu)
    # → không sửa vì có thể là brand name tiếng Anh
    # Ngoại lệ: từ ngắn như "pho", "bun", "com" là ẩm thực VN
    if (
        not _has_vietnamese_mark(token)
        and token[:1].isupper()
        and len(token) > 5
        and re.match(r'^[A-Za-z]+$', token)
    ):
        return True
    return False


def _best_fuzzy_match(norm_phrase: str, token_count: int) -> Optional[Tuple[Tuple[str, ...], float]]:
    if process is None or fuzz is None:
        return None
    _, choices, lookup = _load_places()
    if not choices:
        return None

    # Compare mostly with same-length phrases to avoid over-correction.
    src_tokens = norm_phrase.split()
    filtered = []
    for choice in choices:
        choice_tokens = choice.split()
        if len(choice_tokens) != token_count:
            continue
        # Guard quan trọng: OCR sai dấu/ký tự nhỏ vẫn thường giữ chữ cái đầu từng token.
        # Chặn `binh binh` bị fuzzy thành `binh dinh`.
        if len(choice_tokens) == len(src_tokens):
            if any(s[:1] != c[:1] for s, c in zip(src_tokens, choice_tokens)):
                continue
        filtered.append(choice)
    if not filtered:
        return None

    match = process.extractOne(norm_phrase, filtered, scorer=fuzz.WRatio)
    if not match:
        return None
    matched_key, score, _ = match
    if score < _MIN_FUZZY_SCORE:
        return None
    return lookup[matched_key], float(score)


def _apply_osm_word_corrections(
    original_tokens: List[str],
    stripped_tokens: List[str],
    occupied: Set[int],
) -> Dict[int, str]:
    """
    Áp dụng OSM word dictionary để sửa dấu tiếng Việt cho các token chưa được
    xử lý bởi normalize_place_phrases.
    
    Chiến lược bảo thủ:
    - Chỉ sửa token không có dấu tiếng Việt (hoặc có dấu nhưng sai nhẹ)
    - KHÔNG sửa: ALLCAPS dài, brand names, token đã có đủ dấu VN đặc trưng
    - Ưu tiên n-grams dài hơn đơn token (chính xác hơn khi có ngữ cảnh)
    - Thêm guard: chỉ thay khi canonical có dấu VN đặc trưng (ăm â ê ô ơ ư đ)
      → tránh thay "ba" → "Bà" khi "ba" trong văn cảnh brand/tên riêng
    """
    osm_dict = _load_osm_words()
    if not osm_dict:
        return {}

    replacements: Dict[int, str] = {}
    n_tokens = len(original_tokens)

    # Quét n-grams từ dài đến ngắn
    for n in range(_OSM_MAX_NGRAM, 0, -1):
        for start in range(0, n_tokens - n + 1):
            indexes = list(range(start, start + n))
            # Bỏ qua vị trí đã được xử lý (bởi place gazette hoặc n-gram dài hơn)
            if any(idx in occupied for idx in indexes):
                continue

            raw_toks = [original_tokens[i] for i in indexes]
            stripped_toks = [stripped_tokens[i] for i in indexes]

            # Tạo key tra cứu
            lookup_key = " ".join(stripped_toks)
            canonical_str = osm_dict.get(lookup_key)
            if not canonical_str:
                continue

            # Special case: only correct "so" -> "Số" if it is followed by a number/digit
            if lookup_key == "so":
                if start + 1 < n_tokens:
                    next_tok = original_tokens[start + 1]
                    if not any(ch.isdigit() for ch in next_tok):
                        continue
                else:
                    continue

            # Ambiguous: "quan" can be "Quán" (eatery) or "Quận" (district).
            # Only allow administrative "Quận" in clear admin context, including OSM n-grams.
            if "quan" in stripped_toks and re.search(r'(?i)\bquận\b', canonical_str):
                prev_key = stripped_tokens[start - 1] if start > 0 else ""
                next_tok = original_tokens[start + n] if start + n < n_tokens else ""
                has_admin_context = (
                    prev_key in {"phuong", "ubnd", "quan", "huyen", "tp", "thanh", "pho"}
                    or any(ch.isdigit() for ch in next_tok)
                    or any(any(ch.isdigit() for ch in tok) for tok in raw_toks)
                )
                if not has_admin_context:
                    continue

            # Parse canonical thành tokens
            canonical_tokens = _TOKEN_RE.findall(canonical_str)
            if len(canonical_tokens) != n:
                continue

            # Guard 1: Không sửa token nào trông như brand
            if any(_is_token_brand_like(raw_toks[i]) for i in range(n)):
                continue

            # Guard 2: Phải có cải thiện thực sự (canonical phải có dấu VN đặc trưng)
            # → Tránh thay các từ không dấu sang dạng cũng không dấu
            if not any(_has_shaped_vowel(ct) or _has_vietnamese_mark(ct) for ct in canonical_tokens):
                continue

            # Guard 3: Đối với đơn token — chỉ sửa nếu raw token chưa có dấu đặc trưng
            # (Nếu raw đã có đủ dấu VN đặc trưng thì không cần sửa)
            if n == 1:
                raw_tok = raw_toks[0]
                canonical_tok = canonical_tokens[0]
                # Token đã có dấu Việt là tín hiệu mạnh từ ảnh crop; không đổi sang từ khác.
                if _has_vietnamese_mark(raw_tok):
                    continue
                if _has_shaped_vowel(raw_tok):
                    continue

            # Guard 4: Với n-gram, không ghi đè token đã có dấu Việt bằng token khác.
            if n > 1:
                if any(
                    _has_vietnamese_mark(rt) and rt != ct
                    for rt, ct in zip(raw_toks, canonical_tokens)
                ):
                    continue
                has_improvement = any(
                    not _has_shaped_vowel(rt) and _has_shaped_vowel(ct)
                    for rt, ct in zip(raw_toks, canonical_tokens)
                )
                if not has_improvement:
                    continue

            # Guard 5: Một số base tiếng Việt nhập nhằng, chỉ sửa khi có ngữ cảnh loại POI rõ.
            ambiguous_food = {"pho", "ga", "com", "bun", "bo", "cha", "gia"}
            if any(st in ambiguous_food for st in stripped_toks):
                context = set(stripped_tokens[max(0, start - 3):start] + stripped_tokens[start + n:start + n + 3])
                food_context = {"quan", "nha", "hang", "mon", "mien", "nuong", "an", "cafe", "tiem"}
                if not (context & food_context):
                    continue
            if lookup_key == "trai" and canonical_str.lower() == "trãi":
                continue

            # Áp dụng thay thế
            for off, ct in enumerate(canonical_tokens):
                replacements[start + off] = ct
                occupied.add(start + off)

    return replacements


_CONTEXTUAL_TOKEN_CORRECTIONS = (
    # wrong_base, canonical, required context tokens near it
    ("xe", "Xe", {"may", "moto", "gan", "sua", "cuu", "ho"}),
    ("tho", "Thợ", {"sua", "xe", "cuu", "ho", "hello"}),
    ("vat", "Vặt", {"an", "quan", "mon"}),
    ("mia", "Mía", {"nuoc", "ep", "mia"}),
    ("cung", "Cúng", {"do", "dich", "vu", "tron", "goi"}),
)

_CONTEXTUAL_PHRASE_CORRECTIONS = (
    (("ca", "phe"), ("Cà", "phê")),
    (("cafe",), ("Cafe",)),
    (("nha", "hang"), ("Nhà", "hàng")),
    (("khach", "san"), ("Khách", "sạn")),
    (("nha", "thuoc"), ("Nhà", "thuốc")),
    (("sua", "xe"), ("Sửa", "Xe")),
    (("cuu", "ho", "xe"), ("Cứu", "Hộ", "Xe")),
    (("nuoc", "mia"), ("Nước", "Mía")),
    (("an", "vat"), ("Ăn", "Vặt")),
    (("do", "cung"), ("Đồ", "Cúng")),
)


def _match_case(source: str, canonical: str) -> str:
    """Preserve rough OCR casing while adding Vietnamese spelling."""
    if source.isupper():
        return canonical.upper()
    if source[:1].islower() and not source.isupper():
        return canonical[:1].lower() + canonical[1:]
    return canonical


_SAFE_STANDALONE_ALIASES = {
    "hanoi": "Hà Nội",
    "saigon": "Sài Gòn",
    "hcm": "HCM",
}


def normalize_ocr_spelling(text: str) -> str:
    """
    Context-aware OCR spelling cleanup backed by dictionary rules.

    Conservative design:
    - never changes digit tokens or likely brands/acronyms
    - prefers phrase/context corrections over isolated token guesses
    - preserves token count; junk removal remains in vision.py cleanup
    """
    if not text:
        return text

    matches = list(_TOKEN_RE.finditer(text))
    if not matches:
        return text

    original_tokens = [m.group(0) for m in matches]
    stripped_tokens = [strip_vietnamese_accents(tok) for tok in original_tokens]
    replacements: Dict[int, str] = {}
    occupied: Set[int] = set()

    # Phrase-level common Vietnamese POI words. Exact base only, no fuzzy.
    for src_phrase, dst_phrase in sorted(_CONTEXTUAL_PHRASE_CORRECTIONS, key=lambda item: -len(item[0])):
        n = len(src_phrase)
        for start in range(0, len(original_tokens) - n + 1):
            indexes = range(start, start + n)
            if any(idx in occupied for idx in indexes):
                continue
            if tuple(stripped_tokens[start:start + n]) != src_phrase:
                continue
            raw = original_tokens[start:start + n]
            if any(_is_token_brand_like(tok) for tok in raw):
                continue
            for off, canonical in enumerate(dst_phrase):
                idx = start + off
                replacements[idx] = _match_case(original_tokens[idx], canonical)
                occupied.add(idx)

    # Token-level accent/type corrections only with neighboring context.
    context_window = 3
    for idx, (raw, key) in enumerate(zip(original_tokens, stripped_tokens)):
        if idx in occupied or _is_token_brand_like(raw) or any(ch.isdigit() for ch in raw):
            continue
        left = max(0, idx - context_window)
        right = min(len(stripped_tokens), idx + context_window + 1)
        context = set(stripped_tokens[left:idx] + stripped_tokens[idx + 1:right])
        for wrong_base, canonical, required_context in _CONTEXTUAL_TOKEN_CORRECTIONS:
            if key != wrong_base:
                continue
            if not (context & required_context):
                continue
            # Nếu raw đã có nguyên âm Việt đặc trưng khác base canonical, không đoán.
            if _has_shaped_vowel(raw) and strip_vietnamese_accents(raw) != strip_vietnamese_accents(canonical):
                continue
            replacements[idx] = _match_case(raw, canonical)
            occupied.add(idx)
            break

    # Safe standalone aliases: only when text is short or already has Vietnamese context,
    # not inside longer English brand phrases like `Saigon Post Office`.
    for idx, (raw, key) in enumerate(zip(original_tokens, stripped_tokens)):
        if idx in occupied or _is_token_brand_like(raw):
            continue
        alias = _SAFE_STANDALONE_ALIASES.get(key)
        if not alias:
            continue
        has_vietnamese_context = any(_has_vietnamese_mark(tok) for tok in original_tokens) or len(original_tokens) <= 3
        if not has_vietnamese_context:
            continue
        replacements[idx] = alias
        occupied.add(idx)

    if not replacements:
        return text

    out = []
    last = 0
    for idx, match in enumerate(matches):
        out.append(text[last:match.start()])
        out.append(replacements.get(idx, match.group(0)))
        last = match.end()
    out.append(text[last:])
    return "".join(out)


def normalize_place_phrases(text: str) -> str:
    """Normalize Vietnamese place phrases in OCR text conservatively.

    The function scans token n-grams, longest first, and replaces only complete
    place phrases. Exact accent-stripped matches are preferred; fuzzy matching is
    used only for 2+ token phrases with strong score.
    
    After place-phrase normalization, applies OSM word dictionary corrections
    for remaining tokens that have missing Vietnamese accents.
    """
    if not text:
        return text

    exact, _, _ = _load_places()

    matches = list(_TOKEN_RE.finditer(text))
    if not matches:
        return text

    original_tokens = [m.group(0) for m in matches]
    stripped_tokens = [strip_vietnamese_accents(tok) for tok in original_tokens]
    replacements: Dict[int, str] = {}
    occupied: Set[int] = set()

    # === Guard: If text is predominantly ALLCAPS, skip OSM corrections ===
    # This protects brand names like "AO DAI AND AO BA BA RENTALS"
    alpha_tokens = [t for t in original_tokens if t.isalpha()]
    allcaps_count = sum(1 for t in alpha_tokens if t.isupper() and len(t) >= 2)
    _is_allcaps_text = len(alpha_tokens) > 0 and (allcaps_count / len(alpha_tokens)) >= 0.5

    # === Phase 1: Place phrase gazetteer (vietnam_places.txt) ===
    if exact:
        max_len = min(_MAX_NGRAM, len(matches))
        for n in range(max_len, 0, -1):
            for start in range(0, len(matches) - n + 1):
                indexes = range(start, start + n)
                if any(idx in occupied for idx in indexes):
                    continue

                raw_tokens = tuple(original_tokens[start:start + n])
                if _is_brand_like(raw_tokens):
                    continue

                key = tuple(stripped_tokens[start:start + n])
                canonical = exact.get(key)

                # Single-token fuzzy is too risky for brands; exact only.
                # Nếu OCR đã có dấu Việt, coi đó là tín hiệu mạnh từ ảnh crop; không fuzzy đổi nghĩa.
                if canonical is None and n >= 2 and not any(_has_vietnamese_mark(tok) for tok in raw_tokens):
                    norm_phrase = " ".join(key)
                    fuzzy_match = _best_fuzzy_match(norm_phrase, n)
                    if fuzzy_match:
                        canonical, _ = fuzzy_match

                if canonical is None:
                    continue

                for off, repl in enumerate(canonical):
                    replacements[start + off] = repl
                    occupied.add(start + off)

    # === Phase 2: OSM word dictionary corrections ===
    # Skip for ALLCAPS-dominant text (brand names)
    if not _is_allcaps_text:
        osm_replacements = _apply_osm_word_corrections(
            original_tokens, stripped_tokens, set(occupied)  # pass copy
        )
        # Merge: phase 1 has priority (don't override place gazetteer matches)
        for idx, repl in osm_replacements.items():
            if idx not in occupied:
                replacements[idx] = repl
                occupied.add(idx)

    if not replacements:
        return text

    out = []
    last = 0
    for idx, match in enumerate(matches):
        out.append(text[last:match.start()])
        out.append(replacements.get(idx, match.group(0)))
        last = match.end()
    out.append(text[last:])
    return "".join(out)
