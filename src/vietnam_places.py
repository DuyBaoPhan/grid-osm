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

    Dictionary này được xây dựng từ cache OSM nationwide trong repo.
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
            cleaned: Dict[str, str] = {}
            for key, value in data.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    continue
                key = key.strip().lower()
                value = value.strip()
                if not key or not value:
                    continue
                if key.replace(" ", "").isdigit():
                    continue
                if not _TOKEN_RE.search(value):
                    continue
                cleaned[key] = value
            return cleaned
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
        and len(token) > 6
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


@lru_cache(maxsize=1)
def _load_curated_phrase_corrections() -> Dict[str, str]:
    """Load explicit curated phrase corrections from data/ocr_language_corrections.json."""
    path = _project_root() / "data" / "ocr_language_corrections.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    phrases: Dict[str, str] = {}
    for item in data.get("contextual_phrase_corrections", []):
        if not isinstance(item, dict):
            continue
        source = item.get("source")
        target = item.get("target")
        if not isinstance(source, list) or not isinstance(target, list):
            continue
        source_tokens = [str(tok).strip() for tok in source if str(tok).strip()]
        target_tokens = [str(tok).strip() for tok in target if str(tok).strip()]
        if 2 <= len(source_tokens) <= 5 and len(source_tokens) == len(target_tokens):
            key = " ".join(strip_vietnamese_accents(tok) for tok in source_tokens)
            phrases[key] = " ".join(target_tokens)
    return phrases


@lru_cache(maxsize=1)
def _load_strict_phrase_dictionary() -> Dict[str, Tuple[str, ...]]:
    """
    Build strict 2–5 token dictionary for OCR spelling.

    Sources are already curated/generated dictionaries, but matching is deliberately
    exact on accent-stripped token phrases. This prevents hallucinated corrections,
    word insertion/deletion, and ambiguous single-token guesses.
    """
    phrase_dict: Dict[str, Tuple[str, ...]] = {}

    def add_phrase(phrase: str) -> None:
        canonical_tokens = tuple(m.group(0) for m in _TOKEN_RE.finditer(phrase or ""))
        if not (2 <= len(canonical_tokens) <= 5):
            return
        if any(any(ch.isdigit() for ch in tok) for tok in canonical_tokens):
            return
        if not any(_has_vietnamese_mark(tok) for tok in canonical_tokens):
            return
        key = " ".join(strip_vietnamese_accents(tok) for tok in canonical_tokens)
        if key:
            phrase_dict[key] = canonical_tokens

    exact, _, _ = _load_places()
    for canonical_tokens in exact.values():
        add_phrase(" ".join(canonical_tokens))

    for canonical in _load_osm_words().values():
        add_phrase(canonical)

    for canonical in _load_curated_phrase_corrections().values():
        add_phrase(canonical)

    return phrase_dict


def _is_strict_phrase_blocked(raw_tokens: Tuple[str, ...], canonical_tokens: Tuple[str, ...]) -> bool:
    """Block brand/ALLCAPS/English-like phrases from dictionary rewrite."""
    if len(raw_tokens) != len(canonical_tokens):
        return True
    if len(raw_tokens) < 2 or len(raw_tokens) > 5:
        return True
    if any(any(ch.isdigit() for ch in tok) for tok in raw_tokens):
        return True
    if any(tok.isupper() and len(tok) >= 2 and tok not in {"TP"} for tok in raw_tokens):
        return True
    if any(
        _has_vietnamese_mark(rt)
        and rt != ct
        and strip_vietnamese_accents(rt) != strip_vietnamese_accents(ct)
        for rt, ct in zip(raw_tokens, canonical_tokens)
    ):
        return True

    # Do not rewrite pure English-looking Title Case phrases unless dictionary
    # gives Vietnamese marks in at least one token and source is not all Title Case
    # brand-like words.
    if all(re.fullmatch(r"[A-Za-z]+", tok or "") for tok in raw_tokens):
        long_title_tokens = [tok for tok in raw_tokens if tok[:1].isupper() and len(tok) > 5]
        if long_title_tokens and not any(tok.lower() in {"nguyen", "duong", "buu", "dien", "trung", "tam", "nha", "sach", "pho"} for tok in raw_tokens):
            return True
    return False


def _apply_strict_phrase_corrections(
    original_tokens: List[str],
    stripped_tokens: List[str],
    occupied: Set[int],
) -> Dict[int, str]:
    """Apply only exact accent-stripped 2–5 token phrase corrections."""
    phrase_dict = _load_strict_phrase_dictionary()
    if not phrase_dict:
        return {}

    replacements: Dict[int, str] = {}
    n_tokens = len(original_tokens)
    for n in range(5, 1, -1):
        for start in range(0, n_tokens - n + 1):
            indexes = list(range(start, start + n))
            if any(idx in occupied for idx in indexes):
                continue
            lookup_key = " ".join(stripped_tokens[start:start + n])
            canonical = phrase_dict.get(lookup_key)
            if canonical is None:
                continue
            raw_tokens = tuple(original_tokens[start:start + n])
            if _is_strict_phrase_blocked(raw_tokens, canonical):
                continue
            for off, repl in enumerate(canonical):
                replacements[start + off] = repl
                occupied.add(start + off)
    return replacements


@lru_cache(maxsize=1)
def _load_contextual_token_corrections() -> List[Tuple[str, str, Set[str]]]:
    """Load contextual token corrections dynamically from ocr_language_corrections.json."""
    default_corrections = [
        ("xe", "Xe", {"may", "moto", "gan", "sua", "cuu", "ho"}),
        ("tho", "Thợ", {"sua", "xe", "cuu", "ho", "hello"}),
        ("vat", "Vặt", {"an", "quan", "mon"}),
        ("mia", "Mía", {"nuoc", "ep", "mia"}),
        ("cung", "Cúng", {"do", "dich", "vu", "tron", "goi"}),
        ("lam", "Làm", {"nha", "noi", "bep", "handmade"}),
    ]
    path = _project_root() / "data" / "ocr_language_corrections.json"
    if not path.exists():
        return default_corrections
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        json_corr = data.get("contextual_token_corrections", [])
        if json_corr:
            loaded = []
            for item in json_corr:
                wb = item.get("wrong_base")
                can = item.get("canonical")
                ctx = item.get("context")
                if wb and can and isinstance(ctx, list):
                    loaded.append((wb.strip().lower(), can.strip(), set(str(c).strip().lower() for c in ctx)))
            if loaded:
                return loaded
    except Exception:
        pass
    return default_corrections


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


@lru_cache(maxsize=1)
def _load_safe_standalone_aliases() -> Dict[str, str]:
    """Load safe standalone aliases dynamically from ocr_language_corrections.json."""
    default_aliases = {
        "hanoi": "Hà Nội",
        "saigon": "Sài Gòn",
        "hcm": "HCM",
    }
    path = _project_root() / "data" / "ocr_language_corrections.json"
    if not path.exists():
        return default_aliases
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        json_aliases = data.get("safe_standalone_aliases", {})
        if json_aliases and isinstance(json_aliases, dict):
            return {k.strip().lower(): v.strip() for k, v in json_aliases.items()}
    except Exception:
        pass
    return default_aliases


_KNOWN_TOKENS: Set[str] = set()

def _load_known_tokens() -> Set[str]:
    """Build a comprehensive set of known Vietnamese words/names from gazetteer & OSM words."""
    global _KNOWN_TOKENS
    if _KNOWN_TOKENS:
        return _KNOWN_TOKENS

    tokens = set()
    try:
        # 1. Load from vietnam_places.txt
        exact, _, _ = _load_places()
        for canonical_tokens in exact.values():
            for tok in canonical_tokens:
                tokens.add(strip_vietnamese_accents(tok).lower())

        # 2. Load from osm_words.json
        osm_words = _load_osm_words()
        for canonical in osm_words.values():
            for tok in re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', canonical):
                tokens.add(strip_vietnamese_accents(tok).lower())

        # 3. Load from ocr_language_corrections.json phrase corrections
        curated = _load_curated_phrase_corrections()
        for canonical in curated.values():
            for tok in re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', canonical):
                tokens.add(strip_vietnamese_accents(tok).lower())

        # 4. Load from contextual token corrections
        for _, canonical, _ in _load_contextual_token_corrections():
            for tok in re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', canonical):
                tokens.add(strip_vietnamese_accents(tok).lower())

        # 5. Load from osm_raw_cache_nationwide.json and osm_raw_cache.json if they exist
        for filename in ("osm_raw_cache_nationwide.json", "osm_raw_cache.json"):
            path = _project_root() / "data" / filename
            if path.exists():
                try:
                    with open(path, encoding="utf-8") as f:
                        cache_data = json.load(f)
                    if isinstance(cache_data, list):
                        for item in cache_data:
                            if isinstance(item, str):
                                for tok in re.findall(r'[A-Za-zÀ-ỹĐđ0-9&]+', item):
                                    tokens.add(strip_vietnamese_accents(tok).lower())
                except Exception:
                    pass

    except Exception:
        pass

    # Add very common Vietnamese particles and English/brand map words as safety fallback
    extra_common = {
        "va", "co", "la", "den", "di", "cho", "quan", "bo", "pho", "bun", "com",
        "circle", "arabica", "coffee", "tea", "spa", "gym", "hotel", "restaurant", "cafe", 
        "bar", "pub", "lounge", "shop", "store", "mart", "clinic", "studio", "bank", "atm", 
        "residence", "station", "office", "post", "school", "park", "garden", "center", 
        "plaza", "tower", "building", "mall", "market", "highlands", "starbucks", "passio", 
        "phuc", "long", "cheese", "kfc", "lotteria", "jollibee", "mcdonald", "domino", 
        "pizza", "hut", "subway", "tous", "les", "jours", "paris", "baguette", "givral", 
        "brodard", "abc", "winmart", "coopmart", "lotte", "aeon", "satra", "emart", 
        "seven", "eleven", "gs25", "ministop", "vietcombank", "vietinbank", "bidv", 
        "agribank", "sacombank", "techcombank", "acb", "mbbank", "vpbank", "shb", "vib", 
        "tpb", "ocb", "scb", "hdbank", "eximbank", "nxb"
    }
    tokens.update(extra_common)

    _KNOWN_TOKENS = tokens
    return _KNOWN_TOKENS


def is_known_token(token: str) -> bool:
    """True if token is a standard/known Vietnamese word or place name component."""
    if not token:
        return False
    clean = strip_vietnamese_accents(token).lower()
    return clean in _load_known_tokens()



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
        for wrong_base, canonical, required_context in _load_contextual_token_corrections():
            if key != wrong_base:
                continue
            if not (context & required_context):
                continue
            if raw[:1].islower() and idx > 0 and original_tokens[idx - 1][:1].isupper():
                replacements[idx] = canonical
            else:
                replacements[idx] = _match_case(raw, canonical)
            occupied.add(idx)
            break

    # Safe standalone aliases: only when text is short or already has Vietnamese context,
    # not inside longer English brand phrases like `Saigon Post Office`.
    for idx, (raw, key) in enumerate(zip(original_tokens, stripped_tokens)):
        if idx in occupied or _is_token_brand_like(raw):
            continue
        alias = _load_safe_standalone_aliases().get(key)
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
    """Normalize OCR text using strict exact 2–5 token phrase dictionary matches only."""
    if not text:
        return text

    matches = list(_TOKEN_RE.finditer(text))
    if not matches:
        return text

    original_tokens = [m.group(0) for m in matches]
    stripped_tokens = [strip_vietnamese_accents(tok) for tok in original_tokens]
    replacements: Dict[int, str] = {}
    occupied: Set[int] = set()

    alpha_tokens = [t for t in original_tokens if t.isalpha()]
    allcaps_count = sum(1 for t in alpha_tokens if t.isupper() and len(t) >= 2)
    is_allcaps_text = len(alpha_tokens) > 0 and (allcaps_count / len(alpha_tokens)) >= 0.5

    if not is_allcaps_text:
        replacements.update(_apply_strict_phrase_corrections(original_tokens, stripped_tokens, occupied))

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
