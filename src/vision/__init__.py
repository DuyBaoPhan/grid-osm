# =============================================================
# vision package — public API compatibility layer
# =============================================================

from .detection import extract_pois_from_screenshot
from .rendering import draw_detections, enhance_for_detection
from .geometry import expand_bbox_downward
from .crop_processing import (
    split_crop_into_lines,
    normalize_ocr_background,
    split_crop_into_lines_normalized,
)
from .text_cleaning import (
    _strip_vietnamese_accents,
    _clean_spelling,
    _clean_final_ocr_text,
    _append_missing_known_suffix,
    _is_clean_short_brand_candidate,
    _junk_token_count,
    _looks_like_vietnamese_gibberish,
    _merge_best_diacritics,
    _merge_missing_middle_tokens,
    _merge_overlapping_ocr_continuation,
    _normalized_adds_suspicious_text,
    _normalized_has_valid_main_name_extension,
    _normalized_regresses_quality,
    _score_ocr_text_quality,
    _texts_are_unrelated,
)

__all__ = [
    extract_pois_from_screenshot,
    draw_detections,
    enhance_for_detection,
    expand_bbox_downward,
    split_crop_into_lines,
    normalize_ocr_background,
    split_crop_into_lines_normalized,
    _strip_vietnamese_accents,
    _clean_spelling,
    _clean_final_ocr_text,
    _append_missing_known_suffix,
    _is_clean_short_brand_candidate,
    _junk_token_count,
    _looks_like_vietnamese_gibberish,
    _merge_best_diacritics,
    _merge_missing_middle_tokens,
    _merge_overlapping_ocr_continuation,
    _normalized_adds_suspicious_text,
    _normalized_has_valid_main_name_extension,
    _normalized_regresses_quality,
    _score_ocr_text_quality,
    _texts_are_unrelated,
]
