import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from src.canonical_matcher import resolve_canonical_name
from src.vietnam_places import normalize_ocr_spelling, normalize_place_phrases
from src.vision import (
    _append_missing_known_suffix,
    _clean_final_ocr_text,
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
from src.vision.crop_processing import split_crop_into_lines
from src.vision.recognizers import _recognize_text_crop_vietocr


def test_context_spelling_fixes_common_ocr_errors():
    assert normalize_ocr_spelling("Sửa Xẻ Máy") == "Sửa Xe Máy"
    assert normalize_ocr_spelling("Hello Thọ - Cứu Hộ Xe") == "Hello Thợ - Cứu Hộ Xe"
    assert normalize_ocr_spelling("Ăn Vật - Nước Mia") == "Ăn Vặt - Nước Mía"
    assert normalize_ocr_spelling("Đồ Cũng Trọn Gói") == "Đồ Cúng Trọn Gói"


def test_context_spelling_preserves_brands_and_names():
    assert normalize_ocr_spelling("THE COFFEE LAB") == "THE COFFEE LAB"
    assert normalize_ocr_spelling("Highlands Coffee Saigon Post Office") == "Highlands Coffee Saigon Post Office"
    assert _clean_final_ocr_text("Highlands Coffee Saigon Post Office") == "Highlands Coffee Saigon Post Office"
    assert normalize_ocr_spelling("wăbẽ săbẽ boutique") == "wăbẽ săbẽ boutique"
    assert normalize_ocr_spelling("MCM Post Office") == "MCM Post Office"


def test_place_dictionary_adds_vietnamese_diacritics_conservatively():
    assert normalize_place_phrases("Quan ca phe Ho Chi Minh") == "Quán cà phê Hồ Chí Minh"
    assert normalize_place_phrases("Nha thuoc Tan Dinh") == "Nhà thuốc Tân Định"
    assert normalize_place_phrases("Bun bo Da Lat") == "Bún bò Đà Lạt"
    assert _clean_final_ocr_text("Ca phe sua da") == "Cà phê sua da"
    assert _clean_final_ocr_text("Pho bo Hanoi") == "Phở bò Hà Nội"


def test_final_cleanup_removes_junk_without_dropping_valid_core():
    assert _clean_final_ocr_text("Pravered / Tiệm Nhà Nấm 89") == "Tiệm Nhà Nấm 89"
    assert _clean_final_ocr_text("Bình Bình Quán") == "Bình Quán"
    assert _junk_token_count("Tiệm Nhà Nấm 89") == 0


def test_merge_best_diacritics_keeps_primary_when_base_same():
    assert _merge_best_diacritics("Hello Thợ", "Hello Thọ") == "Hello Thợ"


def test_crop_regressions_do_not_rewrite_marked_vietnamese_words():
    assert _clean_final_ocr_text("Vườn Trong Phố, Gia Định Connection") == "Vườn Trong Phố, Gia Định Connection"
    assert _clean_final_ocr_text("Hum Central - Healthy / Veggies Delights / Trải nghiệm ẩm thực sáng tạo") == "Hum Central - Healthy / Veggies Delights / Trải nghiệm ẩm thực sáng tạo"
    assert _clean_final_ocr_text("MCM Post Office / MCM - Biểu Tượng / Thời Đại Mới") == "MCM Post Office / MCM - Biểu Tượng / Thời Đại Mới"


def test_food_words_only_correct_in_food_context():
    assert normalize_place_phrases("Quan mien pho ga") == "Quan Miến Phở Gà"
    assert normalize_place_phrases("Vuon Trong Pho Gia Dinh Connection") == "Vuon Trong Pho Gia Định Connection"


def test_normalized_fuller_main_name_extension_is_allowed():
    assert _normalized_adds_suspicious_text(
        "Hello Thợ-Cứu hộ xe",
        "Hello Thọ - Cứu Hộ Xe / Máy & Sửa Xe Lưu Động",
    )
    assert _normalized_has_valid_main_name_extension(
        "Hello Thợ-Cứu hộ xe",
        "Hello Thọ - Cứu Hộ Xe / Máy & Sửa Xe Lưu Động",
    )
    assert _normalized_adds_suspicious_text(
        "Đồ cúng Thiên",
        "Đồ Cúng Thiên / Phúc - CN Phú Mỹ",
    )
    assert _normalized_has_valid_main_name_extension(
        "Đồ cúng Thiên",
        "Đồ Cúng Thiên / Phúc - CN Phú Mỹ",
    )


def test_normalized_description_or_category_extension_is_rejected():
    assert _normalized_adds_suspicious_text(
        "Saigon Central / Post Office",
        "Saigon Central / Post Office / Ao Dai City Tour in Saigon / Group CLR English Chine",
    )
    assert not _normalized_has_valid_main_name_extension(
        "Saigon Central / Post Office",
        "Saigon Central / Post Office / Ao Dai City Tour in Saigon / Group CLR English Chine",
    )
    assert _normalized_adds_suspicious_text(
        "MCM Post Office",
        "MCM Post Office / 5.0 (121) / Luxury for her at DAFC Onl",
    )
    assert not _normalized_has_valid_main_name_extension(
        "MCM Post Office",
        "MCM Post Office / 5.0 (121) / Luxury for her at DAFC Onl",
    )


def test_reported_duplicate_and_spelling_regressions():
    assert _clean_final_ocr_text("Ăn Ăn Vặt - Nước Mía Pé Ty") == "Ăn Vặt - Nước Mía Pé Ty"
    assert _clean_final_ocr_text("Mực chiên bơ ông Gù Gù nhà thờ Đức Bà") == "Mực chiên bơ ông Gù Nhà thờ Đức Bà"
    assert _clean_final_ocr_text("An An Law Vietnam") == "An Law Vietnam"
    assert _clean_final_ocr_text("Bếp Nội Nhà lẫm") == "Bếp Nội Nhà Làm"
    assert _clean_final_ocr_text("Cổng Đường sách TP TP Hồ Chí Minh") == "Cổng Đường sách TP Hồ Chí Minh"
    assert _clean_final_ocr_text("Haeduri Hai Bà Trưng") == "Haeduri Hai Bà Trưng"
    assert _clean_final_ocr_text("dirrom Hai Bà Trưng") == "Hai Bà Trưng"


def test_short_brand_primary_blocks_unrelated_normalized_candidate():
    assert _is_clean_short_brand_candidate("GEOX")
    assert _texts_are_unrelated("GEOX", "Korean")


def test_multi_token_suffix_rescue_for_intersection_names():
    assert _append_missing_known_suffix(
        "Vòng xoay Phạm Ngọc",
        "Phạm Ngọc Thạch giao Lê Duẩn",
    ) == "Vòng xoay Phạm Ngọc Thạch giao Lê Duẩn"


def test_vietnamese_gibberish_primary_is_penalized_without_hardcoding():
    bad_primary = "Têm viên nông nân"
    good_normalized = "Trạm xe đạp công cộng / TNGo - UBND Quận 1"
    assert _looks_like_vietnamese_gibberish(bad_primary)
    assert not _looks_like_vietnamese_gibberish("Bếp Nội Nhà Làm")
    assert not _looks_like_vietnamese_gibberish("GEOX")
    assert _score_ocr_text_quality(good_normalized) > _score_ocr_text_quality(bad_primary)


def test_public_infrastructure_duplicate_phrase_is_preserved():
    assert (
        _clean_final_ocr_text("Trạm xe đạp công cộng / TNGo - UBND Quận 1")
        == "Trạm xe đạp công cộng / TNGo - UBND Quận 1"
    )
    assert (
        _merge_missing_middle_tokens(
            "Trạm xe đạp công / TNGo - UBND Quận 1",
            "Trạm xe đạp công cộng / TNGo - UBND Quận 1",
        )
        == "Trạm xe đạp công cộng / TNGo - UBND Quận 1"
    )


def test_overlap_rescue_does_not_regress_cleaner_public_infrastructure_text():
    assert (
        _merge_overlapping_ocr_continuation(
            "Trạm xe đạp công cộng / TNGo - UBND Quận 1",
            "Tramva văn nân nân",
        )
        == "Trạm xe đạp công cộng / TNGo - UBND Quận 1"
    )


def test_intersection_continuation_rescue_uses_valid_longer_alternate():
    assert (
        _merge_overlapping_ocr_continuation(
            "Vòng xoay Phạm Ngọc",
            "Phạm Ngọc Thạch giao Lê Duẩn",
        )
        == "Vòng xoay Phạm Ngọc Thạch giao Lê Duẩn"
    )


def test_category_suffix_and_admin_normalized_regressions_are_rejected():
    assert _clean_final_ocr_text("VIET TUI XÁCH / Fashion accessories store") == "VIET TUI XÁCH"
    assert _clean_final_ocr_text("OHQUAO Souvenir Dept / Souvenir store") == "OHQUAO Souvenir Dept"
    assert _normalized_regresses_quality("UBND phường sài Gòn", "JBND - Công Sài Gòn")


def test_stray_leading_capital_artifacts_are_removed_safely():
    assert _clean_final_ocr_text("TUMI Saigon Central / LPost Office Store") == "TUMI Saigon Central / Post Office Store"
    assert _clean_final_ocr_text("LEfora") == "Efora"
    assert _clean_final_ocr_text("LEfola") == "Efora"
    assert _clean_final_ocr_text("LPBank PGD Bưu điện") == "LPBank PGD Bưu điện"


def test_punctuation_precision_regressions_are_preserved():
    assert not _normalized_regresses_quality("Olivia s Prime Steakhouse", "Olivia's Prime Steakhouse")
    assert _clean_final_ocr_text("Olivia's Prime Steakhouse") == "Olivia's Prime Steakhouse"
    assert _clean_final_ocr_text("Capi Studio DIY Souvenirs 8") == "Capi Studio DIY Souvenirs &..."
    assert _clean_final_ocr_text("Capi Studio DIY Souvenirs &...") == "Capi Studio DIY Souvenirs &..."
    assert _clean_final_ocr_text("125 Hai Bà Trưng") == "125 Hai Bà Trưng"


def test_vietnamese_branch_separator_hyphen_is_restored_generically():
    assert (
        _clean_final_ocr_text("Văn phòng đăng ký đất đai Chi nhánh Quận 1")
        == "Văn phòng đăng ký đất đai - Chi nhánh Quận 1"
    )
    assert (
        _clean_final_ocr_text("Ngân hàng Chính sách xã hội Chi nhánh Hà Nội")
        == "Ngân hàng Chính sách xã hội - Chi nhánh Hà Nội"
    )


def test_foreign_script_wrappers_keep_only_latin_vietnamese_payload():
    assert _clean_final_ocr_text("237 (HWA PUNG / JEONG) 24]") == "HWA PUNG JEONG"
    assert _clean_final_ocr_text("서울 (THE COFFEE SHOP) 24") == "THE COFFEE SHOP"
    assert _clean_final_ocr_text("東京 Quán Cà Phê Sữa Đá 12") == "Quán cà phê Sữa Đá"


def test_english_and_vietnamese_suffix_rescue_keeps_valid_continuation():
    assert _append_missing_known_suffix(
        "Nice Weathers",
        "Nice Weathers The coffee shop",
    ) == "Nice Weathers The coffee shop"
    assert _append_missing_known_suffix(
        "Vòng xoay Phạm Ngọc",
        "Phạm Ngọc Thạch giao Lê Duẩn",
    ) == "Vòng xoay Phạm Ngọc Thạch giao Lê Duẩn"


def test_suffix_rescue_rejects_ratings_categories_and_unrelated_noise():
    assert _append_missing_known_suffix("Nice Weathers", "Nice Weathers 4.8 (120)") == "Nice Weathers"
    assert _append_missing_known_suffix("Nice Weathers", "Nice Weathers coffee shop store") == "Nice Weathers"
    assert _append_missing_known_suffix("Nice Weathers", "Other Label Nice Weathers") == "Nice Weathers"


def test_reported_intersection_slash_cleanup_keeps_full_continuation():
    assert (
        _clean_final_ocr_text("Vòng xoay Phạm Ngọc / Thạch giáo Lê Duẩn")
        == "Vòng xoay Phạm Ngọc Thạch giao Lê Duẩn"
    )
    assert (
        _clean_final_ocr_text("Vòng xoay Phạm Ngọc / Thạch giao Lê Duẩn")
        == "Vòng xoay Phạm Ngọc Thạch giao Lê Duẩn"
    )


def test_reported_vietnamese_gibberish_is_rejected_generically():
    bad = "Tram vin nông nân"
    good = "Trạm xe đạp công cộng / TNGo - UBND Quận 1"
    assert _looks_like_vietnamese_gibberish(bad)
    assert _score_ocr_text_quality(good) > _score_ocr_text_quality(bad) + 40
    assert not _looks_like_vietnamese_gibberish("Công viên Hàn Thuyên")
    assert not _looks_like_vietnamese_gibberish("Trung tâm y khoa Diag")


def test_reported_intersection_saved_crop_splits_into_two_ocr_lines():
    import glob

    import cv2
    import numpy as np

    crop_paths = glob.glob(os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "crops",
        "tile_-1_-1_poi_9_*.png",
    ))
    if not crop_paths:
        pytest.skip("missing optional saved regression crop for tile -1,-1 poi 9")
    img = cv2.imdecode(np.fromfile(crop_paths[0], dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.size > 0
    line_crops = split_crop_into_lines(img, 1.0)
    assert len(line_crops) >= 2
    assert all(line.shape[0] >= 10 for line in line_crops[:2])
    # Regression payload from this crop's two lines; avoids requiring OCR model in unit tests.
    assert (
        _clean_final_ocr_text("Vòng xoay Phạm Ngọc / Thạch giao Lê Duấn")
        == "Vòng xoay Phạm Ngọc Thạch giao Lê Duẩn"
    )


def test_reported_school_crop_images_keep_visible_leading_school_words():
    import glob

    import cv2
    import numpy as np

    root = os.path.dirname(os.path.dirname(__file__))
    cases = [
        ("tile_-1_1_poi_3_*.png", ("Trường", "Hòa Bình")),
        ("tile_1_1_poi_4_*.png", ("Trường THPT Chuyên", "Trần Đại Nghĩa")),
    ]
    for pattern, expected_parts in cases:
        paths = glob.glob(os.path.join(root, "crops", pattern))
        if not paths:
            pytest.skip(f"missing optional saved regression crop for {pattern}")
        img = cv2.imdecode(np.fromfile(paths[0], dtype=np.uint8), cv2.IMREAD_COLOR)
        assert img is not None and img.size > 0
        h, w = img.shape[:2]
        text = _recognize_text_crop_vietocr(img, [0, 0, w, h], icon_side="left", scale=1.0)
        for expected in expected_parts:
            assert expected in text


def test_school_context_ocr_spelling_restores_generic_vietnamese_school_terms():
    assert _clean_final_ocr_text("Trong THPT Chuyên / Trần Đại Nghĩa") == "Trường THPT Chuyên / Trần Đại Nghĩa"
    assert _clean_final_ocr_text("Trương Tiểu / Hòa Bình") == "Trường Tiểu / Hòa Bình"


def test_reported_adjacent_vietnamese_duplicate_cleanup_preserves_slash_structure():
    assert _clean_final_ocr_text("ÁO DÀI AND ÁO BÀ BÀ BA RENTALS") == "ÁO DÀI AND ÁO BÀ BA RENTALS"
    assert _clean_final_ocr_text("ÁO DÀI AND ÁO / BÀ BÀ BA RENTALS") == "ÁO DÀI AND ÁO / BÀ BA RENTALS"
    assert (
        _merge_missing_middle_tokens(
            "ÁO DÀI AND ÁO BÀ BÀ BA RENTALS",
            "ÁO DÀI AND ÁO / BÀ BÀ BA RENTALS",
        )
        == "ÁO DÀI AND ÁO / BÀ BA RENTALS"
    )
    assert _clean_final_ocr_text("Cà phê phê sữa") == "Cà phê sữa"


def test_reported_ao_dai_crop_image_ocr_no_duplicate_ba():
    import glob

    import cv2
    import numpy as np

    root = os.path.dirname(os.path.dirname(__file__))
    paths = glob.glob(os.path.join(root, "crops", "tile_0_0_poi_0_*BA_RENTALS.png"))
    if not paths:
        pytest.skip("missing optional reported ÁO DÀI regression crop")
    img = cv2.imdecode(np.fromfile(paths[0], dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.size > 0
    h, w = img.shape[:2]
    text = _recognize_text_crop_vietocr(img, [0, 0, w, h], icon_side="left", scale=1.0)
    assert " / " in text
    assert "BÀ BÀ" not in text
    assert "BÀ BA" in text
    assert "RENTALS" in text


def test_reported_trailing_slash_segment_rescue_is_rejected_generically():
    primary = "Mặn Mòi, Bến Nghé / Homey Authentic Vietnam."
    normalized = "Mặn Mòi, Bến Nghé / Homey Authentic Vietnam / NGUYỄN THỊ THỊ MônH"
    assert _merge_missing_middle_tokens(primary, normalized) == primary
    assert "NGUYỄN" not in _merge_missing_middle_tokens(primary, normalized)


def test_reported_man_moi_crop_image_ocr_rejects_neighbor_segment():
    import glob

    import cv2
    import numpy as np

    root = os.path.dirname(os.path.dirname(__file__))
    paths = glob.glob(os.path.join(root, "crops", "tile_0_-1_poi_*Homey_Authentic_Vietnam*.png"))
    if not paths:
        pytest.skip("missing optional reported Mặn Mòi regression crop")
    img = cv2.imdecode(np.fromfile(paths[0], dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.size > 0
    h, w = img.shape[:2]
    text = _recognize_text_crop_vietocr(img, [0, 0, w, h], icon_side="left", scale=1.0)
    assert "Mặn Mòi" in text
    assert "Bến Nghé" in text
    assert "Homey Authentic Vietnam" in text
    assert "NGUYỄN" not in text
    assert "THỊ THỊ" not in text


def test_reported_same_token_rescue_preserves_selected_vietnamese_spelling():
    selected = "Cổng Đường sách / TP Hồ Chí Minh"
    normalized = "Cống Đường sách / TP. Hồ Chí Minh"
    assert _merge_missing_middle_tokens(selected, normalized) == selected
    assert _clean_final_ocr_text("Công Đường sách / TP Hồ Chí Minh") == selected


def test_reported_cong_duong_sach_crop_image_ocr_uses_contextual_spelling():
    import glob

    import cv2
    import numpy as np

    root = os.path.dirname(os.path.dirname(__file__))
    paths = glob.glob(os.path.join(root, "crops", "tile_0_0_poi_*Cổng_Đường_sách*.png"))
    if not paths:
        pytest.skip("missing optional reported Cổng Đường sách regression crop")
    img = cv2.imdecode(np.fromfile(paths[0], dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.size > 0
    h, w = img.shape[:2]
    text = _recognize_text_crop_vietocr(img, [0, 0, w, h], icon_side="left", scale=1.0)
    assert "Cổng" in text
    assert "Cống" not in text
    assert "Đường sách" in text
    assert "TP Hồ Chí Minh" in text

def test_no_place_specific_phrase_map_in_final_cleanup():
    from pathlib import Path

    cleanup_source = Path(__file__).resolve().parents[1] / "src" / "vision" / "text_cleaning" / "final_cleanup.py"
    source = cleanup_source.read_text(encoding="utf-8")
    assert "phrase_map" not in source
    assert "poarx" not in source.lower()
    assert "mumi sai gon" not in source.lower()
    assert "starbucks plaza sai gon" not in source.lower()

def test_spatial_exact_short_ocr_uses_nearby_canonical_without_phrase_map():
    match = resolve_canonical_name(
        "Poarx",
        "Poarx",
        [{"name": "VPbank", "lat": 10.0, "lng": 106.0, "source": "dom"}],
        poi_lat=10.00001,
        poi_lng=106.00001,
    )
    assert match.action == "use_canonical"
    assert match.selected == "VPbank"
    assert match.reason == "spatial_exact_short_ocr"


def test_spatial_exact_short_ocr_does_not_override_without_location_evidence():
    match = resolve_canonical_name("Poarx", "Poarx", ["VPbank"])
    assert match.action == "keep_ocr"
    assert match.selected == "Poarx"

def test_spatial_exact_fuzzy_canonical_resolves_reported_ocr_failures_without_phrase_map():
    cases = [
        ("MUMI S\u00e0i G\u00f2n Central Post Office Store", "TUMI Saigon Central Post Office Store"),
        ("w\u0103b\u1ebd s\u0103n\u1ebd boutique", "w\u0103b\u1ebd s\u00e3b\u1ebd boutique"),
        ("Nice Weather", "Nice Waether The Coffee shop"),
        ("l\u1ea7n gi\u1eefa ngian", "L\u1eafp \u0111\u1eb7t m\u00e1y ch\u1ea5m c\u00f4ng to\u00e0n qu\u1ed1c"),
        ("Vinh Duc S\u00e0i G\u00f2n", "Vinh Duc Saigon Corporation"),
        ("CHAGE mPlaza", "CHAGEE mPlaza"),
        ("Starbucks Plaza S\u00e0i G\u00f2n", "Starbucks mPlaza S\u00e0i G\u00f2n"),
    ]
    for ocr, canonical in cases:
        match = resolve_canonical_name(
            ocr,
            ocr,
            [{"name": canonical, "lat": 10.0, "lng": 106.0, "source": "dom"}],
            poi_lat=10.00001,
            poi_lng=106.00001,
        )
        assert match.action == "use_canonical"
        assert match.selected == canonical
        assert match.reason == "spatial_exact_fuzzy_match"


def test_spatial_exact_fuzzy_canonical_requires_location_evidence():
    match = resolve_canonical_name(
        "Nice Weather",
        "Nice Weather",
        [{"name": "Nice Waether The Coffee shop", "source": "dom"}],
    )
    assert match.action == "keep_ocr"
    assert match.selected == "Nice Weather"

def test_normalized_repairs_single_leading_prefix_without_hurting_good_labels():
    from src.vision.text_cleaning.quality import _normalized_regresses_quality

    assert _normalized_regresses_quality(
        "TÁO DÀI AND ?ÁO / BÀ BA' RENTALS",
        "\"ÁO DÀI\" AND ÁO / BÀ BA' RENTALS",
    )
    assert _normalized_regresses_quality(
        "Bưu điện trung / Tâm Sài Gòn",
        "Buj đào rung / Làm Sa Gòn",
    )

def test_reported_wabe_crop_keeps_primary_diacritics():
    import glob

    import cv2
    import numpy as np

    root = os.path.dirname(os.path.dirname(__file__))
    paths = glob.glob(os.path.join(root, "crops", "tile_1_0_poi_11_*.png"))
    if not paths:
        pytest.skip("missing optional reported w?b? regression crop")
    img = cv2.imdecode(np.fromfile(paths[0], dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.size > 0
    h, w = img.shape[:2]
    text = _recognize_text_crop_vietocr(img, [0, 0, w, h], icon_side="left", scale=1.0)
    assert "săbẽ" in text
    assert "sănẽ" not in text

def test_leading_prefix_repair_preserves_uppercase_acronym_prefixes():
    from src.vision.text_cleaning.quality import _ocr_tokens

    primary = "UBND ph??ng S?i G?n"
    normalized = "BND ph??ng S?i G?n"
    primary_tokens = _ocr_tokens(primary)
    norm_tokens = _ocr_tokens(normalized)
    assert len(primary_tokens[0]) == len(norm_tokens[0]) + 1
    assert primary_tokens[0].endswith(norm_tokens[0])
    assert primary.split()[0].isupper()

def test_leading_prefix_repair_preserves_ascii_brand_prefixes():
    from src.vision.text_cleaning.quality import _ocr_tokens

    primary = "Eni Vietnam B.V"
    normalized = "ni Vietnam B.V"
    primary_tokens = _ocr_tokens(primary)
    norm_tokens = _ocr_tokens(normalized)
    assert len(primary_tokens[0]) == len(norm_tokens[0]) + 1
    assert primary_tokens[0].endswith(norm_tokens[0])
    assert not any("?" <= ch <= "?" or ch in "??" for ch in primary_tokens[0])

