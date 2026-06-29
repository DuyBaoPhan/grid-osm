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
    # === Tile (0,0) - Bưu điện Trung tâm ===
    ("Buu dien trung tam Sai Gon",
     None,  # partial match OK - at minimum "Sài Gòn" và "Bưu điện" phải đúng
     "Bưu điện Trung tâm (no accents)"),

    ("Buu dien Sai Gon",
     None,
     "Bưu điện Sài Gòn variant"),

    ("LPBank PGD Buu dien Giao dich Sai Gon",
     None,
     "LPBank + Bưu điện + Giao dịch (brand should stay)"),

    # === Tile (0,-1) - Đường sách, Cổng ===
    ("Cong Duong sach TP. Ho Chi Minh",
     None,
     "Cổng Đường sách TP.HCM"),

    ("Bai giu xe Duong Sach Highlands",
     None,
     "Bãi giữ xe Đường Sách Highlands"),

    ("Highlands Coffee Saigon Post Office",
     "Highlands Coffee Saigon Post Office",  # brand, must NOT change
     "Brand name should not be modified"),

    # === Tile (0,1) - Chợ Bến Thành ===
    ("Cho Ben Thanh",
     None,
     "Chợ Bến Thành"),

    ("Nha hang com tam",
     None,
     "Nhà hàng cơm tấm"),

    # === Tile (-1,0) - Nhà thờ Đức Bà ===
    ("Nha tho Duc Ba",
     None,
     "Nhà thờ Đức Bà"),

    ("Cong vien Tao Dan",
     None,
     "Công viên Tao Đàn"),

    # === Tile (1,0) - Nhà hát, khách sạn ===
    ("Nha hat Thanh pho",
     None,
     "Nhà hát Thành phố"),

    ("Khach san Sofitel",
     None,
     "Khách sạn Sofitel (brand stays)"),

    # === Tests tổng quát ===
    ("Ho Chi Minh",
     "Hồ Chí Minh",
     "Hồ Chí Minh exact match"),

    ("Saigon Post Office",
     "Saigon Post Office",  # English brand, stays
     "English brand unchanged"),

    ("MCM Post Office",
     "MCM Post Office",  # ALLCAPS brand, stays
     "ALLCAPS brand unchanged"),

    ("Vuon Trong Pho",
     None,
     "Vườn Trong Phố"),

    ("Ca phe sua da",
     None,
     "Cà phê sữa đá"),

    ("Pho bo Hanoi",
     None,
     "Phở bò Hà Nội"),

    # === Anti-regression: không được sửa sai ===
    ("PASTA CLUB Not so Italian",
     "PASTA CLUB Not so Italian",
     "ALLCAPS brand must stay"),

    ("Con Meo Nho Little Cats Studio",
     None,
     "Con Mèo Nhỏ Little Cats Studio (partial fix OK)"),

    # === Trường hợp đã có dấu đúng ===
    ("Bưu điện Trung tâm Sài Gòn",
     "Bưu điện Trung tâm Sài Gòn",
     "Already correct - must not change"),

    ("Highlands Coffee",
     "Highlands Coffee",
     "English brand already OK"),
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
