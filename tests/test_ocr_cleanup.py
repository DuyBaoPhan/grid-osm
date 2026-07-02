import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

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
    assert crop_paths, "missing saved regression crop for tile -1,-1 poi 9"
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
