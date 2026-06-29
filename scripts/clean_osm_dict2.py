"""
Lần 2 cleanup: loại bỏ các từ quá ambiguous không thể sửa an toàn ở đơn token.
"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open('data/osm_words.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

print(f"Before: {len(d)} entries")

# === Từ 3 ký tự có hai nghĩa rất khác nhau trong VN ===
# Không thể sửa đúng ở đơn token, chỉ đúng trong bigram/trigram
# (đã có trong bigram context trong dict, bigrams an toàn hơn)
ambiguous_3char = {
    'con',   # Con (con mèo) vs Côn (Côn Đảo) - chọn sai
    'coi',   # Coi (xem) vs Côi (côi cút) 
    'xom',   # Xóm (village) vs Xốm - context dependent
    'bai',   # Bãi (beach/lot) vs Bài (lesson) - hai nghĩa quan trọng
    'giu',   # Giữ (keep) OK but context-sensitive
    'dua',   # Dưa (watermelon) vs Đưa (give) vs Dúa - ambiguous
    'lua',   # Lúa (rice) vs Lửa (fire) vs Lua - all different
    'sua',   # Sữa (milk) vs Sửa (fix) vs Sưa - ambiguous
    'phu',   # Phủ (government) vs Phụ (secondary) - different meanings
    'noi',   # Nội (inner) vs Nơi (place) vs Nói (speak) - all different
    'mui',   # Mùi (smell) vs Múi (segment) vs Muỉ - ambiguous
    'ngu',   # Ngủ (sleep) vs Ngũ (five) vs Ngư (fishing) - ambiguous
    'tai',   # Tài (talent) vs Tai (ear) vs Tái (pale) - ambiguous
    'thu',   # Thu (autumn) vs Thư (letter) vs Thủ - ambiguous
    'thu',   # duplicate
    'ky',    # Kỳ (odd) vs Ký (sign) vs Kỷ (decade) - ambiguous
    'lam',   # Lam (blue) vs Làm (do) vs Lam - ambiguous
    'an',    # Ăn (eat) - already removed but check
    'van',   # Vàn (?) vs Vạn (ten thousand) vs Văn (culture) - ambiguous
}

removed = []
for key in list(d.keys()):
    if key in ambiguous_3char:
        print(f"  Remove ambiguous: {key!r} -> {d[key]!r}")
        del d[key]
        removed.append(key)

# Cũng loại 'pho' vì phố vs phở vs Pho (English)
for key in ['pho']:
    if key in d:
        print(f"  Remove ambiguous: {key!r} -> {d[key]!r}")
        del d[key]
        removed.append(key)

print(f"\nRemoved {len(removed)} entries")
print(f"After: {len(d)} entries")

with open('data/osm_words.json', 'w', encoding='utf-8') as f:
    json.dump(d, f, ensure_ascii=False, indent=2, sort_keys=True)

print("Saved osm_words.json")

# Verify key entries still exist  
check = ['buu dien', 'duong sach', 'bai giu xe', 'vuon', 'nha hang', 'ca phe',
         'dong khoi', 'le loi', 'ngan hang', 'benh vien', 'cong vien', 'sieu thi',
         'nha tho', 'cho', 'truong', 'khach san', 'trung tam', 'cua hang']
print("\nVerifying key entries:")
for k in check:
    v = d.get(k)
    status = '✓' if v else '✗'
    val_str = repr(v) if v else 'NOT FOUND'
    print(f"  {status} {k!r} -> {val_str}")
