# Conservative SymSpell OCR corrections for Vietnamese POI names.

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable

try:
    from symspellpy import SymSpell, Verbosity
except Exception:  # pragma: no cover - optional dependency guard
    SymSpell = None
    Verbosity = None

try:
    import config
except Exception:  # pragma: no cover
    config = None

from .dictionary import _strip_vietnamese_accents

_TOKEN_RE = re.compile(r"[A-Za-zÀ-ỹĐđ0-9]+")
_VIET_ANY_RE = re.compile(r"[À-ỹĐđ]")
_ALLOWED_ALLCAPS = {"TP"}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _enabled() -> bool:
    return bool(getattr(config, "OCR_SYMSPELL_ENABLED", True))


def _max_edit_distance() -> int:
    return int(getattr(config, "OCR_SYMSPELL_MAX_EDIT_DISTANCE", 1))


def _min_term_count() -> int:
    return int(getattr(config, "OCR_SYMSPELL_MIN_TERM_COUNT", 2))


def _iter_json_values(path: Path) -> Iterable[str]:
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, str):
                yield value
    elif isinstance(data, list):
        for value in data:
            if isinstance(value, str):
                yield value


def _iter_correction_targets(path: Path) -> Iterable[str]:
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    for item in data.get("contextual_phrase_corrections", []):
        target = item.get("target") if isinstance(item, dict) else None
        if isinstance(target, list):
            yield " ".join(str(tok).strip() for tok in target if str(tok).strip())
    for item in data.get("contextual_token_corrections", []):
        canonical = item.get("canonical") if isinstance(item, dict) else None
        if isinstance(canonical, str):
            yield canonical
    aliases = data.get("safe_standalone_aliases", {})
    if isinstance(aliases, dict):
        for value in aliases.values():
            if isinstance(value, str):
                yield value


def _looks_like_brand_or_unsafe(token: str) -> bool:
    if not token:
        return True
    if any(ch.isdigit() for ch in token):
        return True
    if token.isupper() and len(token) >= 2 and token not in _ALLOWED_ALLCAPS:
        return True
    if len(token) < 3:
        return True
    return False


def _is_safe_dictionary_token(token: str) -> bool:
    if _looks_like_brand_or_unsafe(token):
        return False
    if not re.fullmatch(r"[A-Za-zÀ-ỹĐđ]+", token):
        return False
    # Avoid pure long English/brand words from OSM names. Vietnamese-marked tokens
    # and common short service words remain useful for accent restoration.
    if not _VIET_ANY_RE.search(token) and token[:1].isupper() and len(token) > 6:
        return False
    return True


def _match_case(source: str, canonical: str) -> str:
    if source.islower():
        return canonical[:1].lower() + canonical[1:]
    if source[:1].isupper():
        return canonical[:1].upper() + canonical[1:]
    return canonical


@lru_cache(maxsize=1)
def _load_term_map() -> dict[str, str]:
    root = _project_root()
    phrases: list[str] = []

    places = root / "data" / "vietnam_places.txt"
    if places.exists():
        try:
            phrases.extend(
                line.strip()
                for line in places.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")
            )
        except Exception:
            pass

    phrases.extend(_iter_json_values(root / "data" / "osm_words.json") or [])
    phrases.extend(_iter_correction_targets(root / "data" / "ocr_language_corrections.json") or [])

    counts: dict[str, int] = {}
    canonical: dict[str, str] = {}
    for phrase in phrases:
        for token in _TOKEN_RE.findall(phrase or ""):
            if not _is_safe_dictionary_token(token):
                continue
            key = _strip_vietnamese_accents(token).lower()
            if not key or key.isdigit():
                continue
            counts[key] = counts.get(key, 0) + 1
            # Prefer a marked Vietnamese spelling when available.
            old = canonical.get(key)
            if old is None or (_VIET_ANY_RE.search(token) and not _VIET_ANY_RE.search(old)):
                canonical[key] = token

    return {
        key: value
        for key, value in canonical.items()
        if counts.get(key, 0) >= _min_term_count()
    }


@lru_cache(maxsize=1)
def _load_phrase_map() -> dict[tuple[str, ...], tuple[str, ...]]:
    root = _project_root()
    phrases: list[str] = []

    places = root / "data" / "vietnam_places.txt"
    if places.exists():
        try:
            phrases.extend(
                line.strip()
                for line in places.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")
            )
        except Exception:
            pass

    phrases.extend(_iter_json_values(root / "data" / "osm_words.json") or [])
    phrases.extend(_iter_correction_targets(root / "data" / "ocr_language_corrections.json") or [])

    phrase_map: dict[tuple[str, ...], tuple[str, ...]] = {}
    for phrase in phrases:
        tokens = tuple(_TOKEN_RE.findall(phrase or ""))
        if not (2 <= len(tokens) <= 5):
            continue
        if any(any(ch.isdigit() for ch in tok) for tok in tokens):
            continue
        if any(tok.isupper() and len(tok) >= 2 and tok not in _ALLOWED_ALLCAPS for tok in tokens):
            continue
        if not any(_VIET_ANY_RE.search(tok) for tok in tokens):
            continue
        key = tuple(_strip_vietnamese_accents(tok).lower() for tok in tokens)
        phrase_map[key] = tokens
    return phrase_map


@lru_cache(maxsize=1)
def _build_symspell():
    if SymSpell is None:
        return None
    max_dist = _max_edit_distance()
    sym = SymSpell(max_dictionary_edit_distance=max_dist, prefix_length=7)
    for key in _load_term_map().keys():
        sym.create_dictionary_entry(key, 1)
    return sym


def apply_symspell_ocr_corrections(text: str) -> str:
    """Apply conservative token-level SymSpell corrections to OCR text."""
    if not text or not _enabled():
        return text
    sym = _build_symspell()
    if sym is None:
        return text
    term_map = _load_term_map()
    if not term_map:
        return text

    replacements: dict[int, str] = {}
    matches = list(_TOKEN_RE.finditer(text))

    phrase_map = _load_phrase_map()
    occupied: set[int] = set()
    n_tokens = len(matches)
    for n in range(min(5, n_tokens), 1, -1):
        for start in range(0, n_tokens - n + 1):
            indexes = range(start, start + n)
            if any(idx in occupied for idx in indexes):
                continue
            raw_tokens = [matches[idx].group(0) for idx in indexes]
            if any(_looks_like_brand_or_unsafe(tok) for tok in raw_tokens):
                continue
            if any(_VIET_ANY_RE.search(tok) for tok in raw_tokens):
                continue
            key = tuple(_strip_vietnamese_accents(tok).lower() for tok in raw_tokens)
            canonical_tokens = phrase_map.get(key)
            if not canonical_tokens:
                continue
            for off, canonical in enumerate(canonical_tokens):
                replacements[start + off] = _match_case(raw_tokens[off], canonical)
                occupied.add(start + off)

    # Do not apply single-token accent restoration. Even exact base matches are
    # ambiguous in POI names (`Hai` vs `Hải`, `giao` vs `giáo`, `Minh` vs
    # `Mình`). Keep corrections at phrase level where context disambiguates.

    if not replacements:
        return text

    out: list[str] = []
    last = 0
    for idx, match in enumerate(matches):
        out.append(text[last:match.start()])
        out.append(replacements.get(idx, match.group(0)))
        last = match.end()
    out.append(text[last:])
    return "".join(out)
