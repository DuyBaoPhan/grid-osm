# OCR quality helpers re-exported from text_cleaning.py.

from .text_cleaning import (
    _looks_like_bad_ocr,
    _looks_like_vietnamese_gibberish,
    _score_ocr_text_quality,
    _ocr_artifact_score,
    _normalized_regresses_quality,
)

__all__ = [
    "_looks_like_bad_ocr",
    "_looks_like_vietnamese_gibberish",
    "_score_ocr_text_quality",
    "_ocr_artifact_score",
    "_normalized_regresses_quality",
]
