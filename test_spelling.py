"""
test_spelling.py — Kiểm tra pipeline sửa dấu tiếng Việt cho OCR

Test cases từ 5 tile đầu tiên:
  Tile (0,0):   Bưu điện Trung tâm Sài Gòn (tâm quét)
  Tile (0,-1):  Phía Bắc - Đường Sách, Cao ốc văn phòng
  Tile (0,1):   Phía Nam - Bến Thành, chợ, cửa hàng
  Tile (-1,0):  Phía Tây - Nhà thờ Đức Bà, công viên
  Tile (1,0):   Phía Đông - Nhà hát Thành phố, khách sạn
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src.vietnam_places import normalize_place_phrases

# ============================================================
# Test cases: (input_ocr, expected_output, description)
# ============================================================
test_cases = [
    ("Nguyen Van Binh", "Nguyễn Văn Bình", "Exact 3-token person/street name"),
    ("Buu dien trung tam", "Bưu điện Trung tâm", "Exact 4-token POI phrase"),
    ("duong sach", "Đường sách", "Exact 2-token phrase"),
    ("Nha sach Kim Dong", "Nhà sách Kim Đồng", "Exact 4-token bookstore name"),
    ("pho", "pho", "Ambiguous single token must stay unchanged"),
    ("mai", "mai", "Ambiguous single token must stay unchanged"),
    ("ga", "ga", "Ambiguous single token must stay unchanged"),
    ("pho ga", "Phở Gà", "Ambiguous words corrected only in clear phrase"),
    ("pho nguyen hue", "Phố Nguyễn Huệ", "Ambiguous pho corrected by clear street phrase"),
    ("PASTA CLUB Not so Italian", "PASTA CLUB Not so Italian", "ALLCAPS brand must stay"),
    ("MCM Post Office", "MCM Post Office", "ALLCAPS brand unchanged"),
    ("Highlands Coffee Saigon Post Office", "Highlands Coffee Saigon Post Office", "English brand phrase unchanged"),
    ("Bưu điện Trung tâm Sài Gòn", "Bưu điện Trung tâm Sài Gòn", "Already correct - must not change"),
    ("Highlands Coffee", "Highlands Coffee", "English brand already OK"),
]


def check_improvement(original: str, result: str) -> bool:
    """True nếu result cải thiện so với original (thêm dấu tiếng Việt)."""
    from src.vietnam_places import _has_shaped_vowel, _TOKEN_RE
    orig_viet = sum(1 for m in _TOKEN_RE.finditer(original) if _has_shaped_vowel(m.group()))
    res_viet = sum(1 for m in _TOKEN_RE.finditer(result) if _has_shaped_vowel(m.group()))
    return res_viet >= orig_viet  # ít nhất không làm tệ hơn


def run_tests():
    passed = 0
    failed = 0
    improved = 0

    print("=" * 70)
    print("Vietnamese OCR Spell Correction - Test Suite")
    print("=" * 70)

    for i, (input_text, expected, description) in enumerate(test_cases, 1):
        result = normalize_place_phrases(input_text)
        
        if expected is not None:
            # Exact match required
            ok = (result == expected)
            if ok:
                passed += 1
                icon = "✓"
            else:
                failed += 1
                icon = "✗"
            print(f"\n{icon} [{i:2d}] {description}")
            print(f"       Input:    {input_text!r}")
            if not ok:
                print(f"       Got:      {result!r}")
                print(f"       Expected: {expected!r}")
        else:
            # No exact expectation — check that result is at least as good
            got_better = check_improvement(input_text, result)
            changed = (result != input_text)
            if got_better:
                passed += 1
                icon = "✓" if changed else "~"
                if changed:
                    improved += 1
            else:
                failed += 1
                icon = "✗"
            print(f"\n{icon} [{i:2d}] {description}")
            print(f"       Input:  {input_text!r}")
            if changed:
                print(f"       Result: {result!r}")
            else:
                print(f"       (unchanged)")
            if not got_better:
                print(f"       WARNING: Result may have regressed!")

    print("\n" + "=" * 70)
    total = len(test_cases)
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"         {improved} inputs were improved (got Vietnamese accents)")
    print("=" * 70)
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
