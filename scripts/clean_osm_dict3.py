"""
Cleanup round 3: Loại bỏ các từ 3-4 ký tự quá ambiguous giữa tiếng Việt và Hán-Việt
"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open('data/osm_words.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

print(f"Before: {len(d)} entries")

# Từ Hán-Việt thường dùng trong tên riêng, KHÔNG nên tự động thêm dấu
# vì chúng có nhiều dạng (Gia/Giá, Dinh/Định/Đình, Hoa/Họa, Long/Lòng...)
han_viet_ambiguous = {
    'gia',   # Gia (Gia Lai, gia đình) vs Giá (giá cả)
    'dinh',  # Định (Gia Định) vs Đình (đình làng) vs Dinh (dinh thự)
    'tinh',  # Tĩnh vs Tình vs Tỉnh vs Tinh
    'binh',  # Bình (binh lính) vs Bính vs Bình (yên bình)
    'hung',  # Hùng vs Hung vs Hưng
    'quoc',  # Quốc vs Quóc
    'thi',   # Thị (thị trường) vs Thì (thì giờ) vs Thi (thi cử)
    'thanh', # Thành (thành phố) vs Thanh (màu) - actually OK: Thành is standard
    'minh',  # Minh (sáng) - single ambiguous
    'phat',  # Phát (phát triển) vs Phật (Phật giáo)
    'long',  # Long (rồng) vs Lòng (trái tim) - different
    'hoa',   # Hoa (bông hoa) vs Họa (họa sĩ) vs Hóa (hóa học)
    'tam',   # Tâm (trái tim) vs Tám (số 8) vs Tám (tâm sự)
    'tuan',  # Tuần (week) vs Tuấn (tên)
    'hung',  # Hùng vs Hưng
    'cuong', # Cường vs Cưỡng
    'sang',  # Sáng vs Sang
    'nhan',  # Nhân vs Nhận
    'bach',  # Bạch (white) vs Bach
    'duc',   # Đức vs Dục
    'linh',  # Linh vs Lĩnh
    'dat',   # Đạt vs Đất vs Đặt
    'ky',    # already removed
    'khu',   # Khu (area) - actually fine
    'loi',   # Lợi vs Lời vs Lối
    'dung',  # Dũng vs Dùng vs Đúng
    'tung',  # Từng vs Tùng vs Túng
}

removed = []
for key in list(d.keys()):
    if key in han_viet_ambiguous:
        val = d[key]
        print(f"  Remove: {key!r} -> {val!r}")
        del d[key]
        removed.append(key)

print(f"\nRemoved {len(removed)} entries")
print(f"After: {len(d)} entries")

with open('data/osm_words.json', 'w', encoding='utf-8') as f:
    json.dump(d, f, ensure_ascii=False, indent=2, sort_keys=True)

print("Saved.")

# Verify the critical ones still exist
check_critical = ['buu dien', 'duong sach', 'bai giu xe', 'vuon', 'nha hang', 
                  'ca phe', 'ngan hang', 'benh vien', 'cong vien', 'sieu thi',
                  'nha tho', 'cho', 'truong', 'khach san', 'trung tam']
print("\nCritical entries check:")
all_ok = True
for k in check_critical:
    v = d.get(k)
    status = 'OK' if v else 'MISSING!'
    val = repr(v) if v else 'NOT FOUND'
    print(f"  {status}: {k!r} -> {val}")
    if not v:
        all_ok = False
print("All critical entries:", "OK" if all_ok else "SOME MISSING!")
