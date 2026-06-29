import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from src.vietnam_places import normalize_ocr_spelling, normalize_place_phrases
from src.vision import _clean_final_ocr_text, _junk_token_count, _merge_best_diacritics


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
    assert normalize_place_phrases("Nha thuoc Sa Dec") == "Nhà thuốc Sa Đéc"
    assert normalize_place_phrases("Bun bo Da Lat") == "Bún bò Đà Lạt"
    assert _clean_final_ocr_text("Ca phe sua da") == "Cà phê sữa đá"
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
    assert normalize_place_phrases("Quan mien pho ga") == "Quan Miền Phở gà"
    assert normalize_place_phrases("Vuon Trong Pho Gia Dinh Connection") == "Vườn Trong Phố Gia Định Connection"
