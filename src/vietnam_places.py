"""Vietnam place-name normalization for OCR post-processing.

This module keeps the place gazetteer outside vision.py and applies conservative
phrase-level correction only. It is designed to fix OCR accent/noise errors in
Vietnamese place names without rewriting brand names.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from rapidfuzz import fuzz, process
except Exception:  # pragma: no cover - optional runtime dependency fallback
    fuzz = None
    process = None


_TOKEN_RE = re.compile(r"[A-Za-zÀ-ỹĐđ0-9]+")
_MAX_NGRAM = 5
_MIN_FUZZY_SCORE = 92


_ACCENT_MAP = (
    (r"[àáảãạăằắẳẵặâầấẩẫậ]", "a"),
    (r"[èéẻẽẹêềếểễệ]", "e"),
    (r"[ìíỉĩị]", "i"),
    (r"[òóỏõọôồốổỗộơờớởỡợ]", "o"),
    (r"[ùúủũụưừứửữự]", "u"),
    (r"[ỳýỷỹỵ]", "y"),
    (r"[đ]", "d"),
)


def strip_vietnamese_accents(text: str) -> str:
    """Strip Vietnamese accents and lowercase for stable matching."""
    s = (text or "").lower()
    for pattern, repl in _ACCENT_MAP:
        s = re.sub(pattern, repl, s)
    return s


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


def normalize_place_phrases(text: str) -> str:
    """Normalize Vietnamese place phrases in OCR text conservatively.

    The function scans token n-grams, longest first, and replaces only complete
    place phrases. Exact accent-stripped matches are preferred; fuzzy matching is
    used only for 2+ token phrases with strong score.
    """
    if not text:
        return text

    exact, _, _ = _load_places()
    if not exact:
        return text

    matches = list(_TOKEN_RE.finditer(text))
    if not matches:
        return text

    original_tokens = [m.group(0) for m in matches]
    stripped_tokens = [strip_vietnamese_accents(tok) for tok in original_tokens]
    replacements: Dict[int, str] = {}
    occupied = set()

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
            if canonical is None and n >= 2:
                norm_phrase = " ".join(key)
                fuzzy_match = _best_fuzzy_match(norm_phrase, n)
                if fuzzy_match:
                    canonical, _ = fuzzy_match

            if canonical is None:
                continue

            for off, repl in enumerate(canonical):
                replacements[start + off] = repl
                occupied.add(start + off)

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
