"""Canonical name matching for OCR POI text.

Pipeline:
OCR text -> conservative cleanup -> compare with nearby/browser/OSM names ->
use canonical only when confidence is high; otherwise keep OCR or flag review.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Dict, Iterable, List, Optional

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover
    fuzz = None

try:
    from src.vietnam_places import strip_vietnamese_accents
except ImportError:  # pragma: no cover
    from vietnam_places import strip_vietnamese_accents

_TOKEN_RE = re.compile(r"[A-Za-zÀ-ỹĐđ0-9]+")
_VIET_MARK_RE = re.compile(r"[À-ỹĐđ]")


@dataclass(frozen=True)
class CanonicalMatch:
    original: str
    cleaned: str
    selected: str
    source: str
    score: float
    action: str
    needs_review: bool
    reason: str


def _tokens(text: str) -> List[str]:
    return [strip_vietnamese_accents(t).lower() for t in _TOKEN_RE.findall(text or "")]


def _token_f1(a: Iterable[str], b: Iterable[str]) -> float:
    left = list(a)
    right = list(b)
    if not left or not right:
        return 0.0
    remaining = right.copy()
    overlap = 0
    for tok in left:
        if tok in remaining:
            remaining.remove(tok)
            overlap += 1
    precision = overlap / max(1, len(left))
    recall = overlap / max(1, len(right))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _char_score(a: str, b: str) -> float:
    aa = " ".join(_tokens(a))
    bb = " ".join(_tokens(b))
    if not aa or not bb:
        return 0.0
    if fuzz is not None:
        return float(fuzz.WRatio(aa, bb)) / 100.0
    # Small fallback: token F1 only.
    return _token_f1(aa.split(), bb.split())


def _has_vietnamese_mark(text: str) -> bool:
    return bool(_VIET_MARK_RE.search(text or ""))


def _is_brand_sensitive(text: str) -> bool:
    toks = _TOKEN_RE.findall(text or "")
    if not toks:
        return False
    allcaps = sum(1 for t in toks if t.isupper() and len(t) >= 2)
    english_title = sum(1 for t in toks if t[:1].isupper() and not _has_vietnamese_mark(t) and len(t) >= 5)
    return allcaps > 0 or english_title >= 2


def _token_delta_safe(ocr: str, canonical: str) -> bool:
    ot = _tokens(ocr)
    ct = _tokens(canonical)
    if not ot or not ct:
        return False
    if abs(len(ot) - len(ct)) > 2:
        return False
    # If OCR is a clean subset of a longer canonical with at most 2 missing descriptive tokens, allow.
    if len(ct) > len(ot):
        missing = len(ct) - len(ot)
        return missing <= 2 and _token_f1(ot, ct) >= 0.86
    return True


def _distance_score(candidate: Dict[str, Any], poi_lat: Optional[float], poi_lng: Optional[float]) -> float:
    if poi_lat is None or poi_lng is None:
        return 0.0
    lat = candidate.get("lat")
    lng = candidate.get("lng")
    if lat is None or lng is None:
        return 0.0
    try:
        # Approx meters, enough for tie-breaks inside one tile.
        dy = (float(lat) - float(poi_lat)) * 111_320.0
        dx = (float(lng) - float(poi_lng)) * 111_320.0 * math.cos(math.radians(float(poi_lat)))
        dist_m = math.sqrt(dx * dx + dy * dy)
    except Exception:
        return 0.0
    if dist_m <= 15:
        return 0.05
    if dist_m <= 40:
        return 0.03
    if dist_m <= 80:
        return 0.01
    return -0.04


def _iter_nearby_names(nearby_names: Any) -> List[Dict[str, Any]]:
    if not nearby_names:
        return []
    out: List[Dict[str, Any]] = []
    if isinstance(nearby_names, dict):
        iterable = nearby_names.items()
        for name, meta in iterable:
            item = {"name": str(name), "source": "nearby"}
            if isinstance(meta, dict):
                item.update(meta)
            out.append(item)
    else:
        for entry in nearby_names:
            if isinstance(entry, str):
                out.append({"name": entry, "source": "nearby"})
            elif isinstance(entry, dict) and entry.get("name"):
                out.append(entry.copy())
    # Deduplicate names preserving first source.
    seen = set()
    deduped = []
    for item in out:
        key = " ".join(_tokens(item.get("name", "")))
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def resolve_canonical_name(
    ocr_text: str,
    cleaned_text: Optional[str] = None,
    nearby_names: Any = None,
    *,
    poi_lat: Optional[float] = None,
    poi_lng: Optional[float] = None,
    min_accept_score: float = 0.94,
    min_review_score: float = 0.86,
) -> CanonicalMatch:
    """Resolve final POI name using conservative OCR + nearby canonical names.

    High-confidence nearby match -> canonical.
    No high match + clean OCR -> OCR.
    Ambiguous/dirty/near miss -> OCR with needs_review=True.
    """
    original = (ocr_text or "").strip()
    cleaned = (cleaned_text if cleaned_text is not None else original).strip()
    if not cleaned:
        return CanonicalMatch(original, cleaned, "", "ocr", 0.0, "empty", True, "empty_ocr")

    ocr_tokens = _tokens(cleaned)
    if not ocr_tokens:
        return CanonicalMatch(original, cleaned, cleaned, "ocr", 0.0, "keep_ocr", True, "no_tokens")

    candidates = _iter_nearby_names(nearby_names)
    best_item: Optional[Dict[str, Any]] = None
    best_score = 0.0
    best_text = ""
    for item in candidates:
        cand = str(item.get("name", "")).strip()
        if not cand:
            continue
        token_score = _token_f1(ocr_tokens, _tokens(cand))
        char_score = _char_score(cleaned, cand)
        score = 0.62 * char_score + 0.38 * token_score + _distance_score(item, poi_lat, poi_lng)
        score = max(0.0, min(1.0, score))
        if score > best_score:
            best_score = score
            best_item = item
            best_text = cand

    if best_item and best_score >= min_accept_score and _token_delta_safe(cleaned, best_text):
        # Guard brand names: require very high score before replacing brand-sensitive OCR.
        if _is_brand_sensitive(cleaned) and best_score < 0.975:
            return CanonicalMatch(original, cleaned, cleaned, "ocr", best_score, "keep_ocr", True, "brand_sensitive_near_match")
        return CanonicalMatch(
            original,
            cleaned,
            best_text,
            str(best_item.get("source", "nearby")),
            best_score,
            "use_canonical",
            False,
            "high_confidence_nearby_match",
        )

    # Clean OCR fallback. Flag if close to nearby but below safe threshold.
    review = False
    reason = "clean_ocr_no_nearby_match"
    if best_item and best_score >= min_review_score:
        review = True
        reason = "nearby_match_below_accept_threshold"
    if len(ocr_tokens) <= 1 and len(cleaned) <= 4:
        review = True
        reason = "short_ocr"
    if re.search(r"[?#*]", cleaned):
        review = True
        reason = "ocr_artifact_chars"

    return CanonicalMatch(original, cleaned, cleaned, "ocr", best_score, "keep_ocr", review, reason)
