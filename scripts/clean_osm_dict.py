"""
Sửa các vấn đề trong osm_words.json:
1. Loại bỏ từ ngắn tiếng Anh bị nhầm thành tiếng Việt (so, no, to, the, in, on...)
2. Sửa trung -> Trưng (sai: trung=trung, truong=trường, trung≠trưng)
3. Sửa trong -> Trọng (có thể sai trong "vườn trong phố")
4. Sửa trang -> Tráng (có thể OK)
"""
import json, sys, re
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Load
with open('data/osm_words.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

print(f"Before: {len(d)} entries")

# === Danh sách từ ngắn cần loại bỏ vì ambiguous với tiếng Anh ===
# Quy tắc: từ ngắn (<=4 ký tự) không có ký tự đặc trưng Việt trong base
# và khả năng cao là tiếng Anh thông thường
english_common_short = {
    # Articles/prepositions/conjunctions tiếng Anh
    'so', 'no', 'to', 'the', 'in', 'on', 'at', 'of', 'or', 'an',
    'be', 'do', 'go', 'up', 'us', 'by', 'my', 'we',
    # Từ 3 chữ cái phổ biến tiếng Anh có thể bị map sang VN
    'not', 'but', 'for', 'are', 'was', 'has', 'had', 'its', 'our',
    'out', 'all', 'one', 'can', 'may', 'him', 'his', 'her', 'the',
    'any', 'old', 'new', 'big', 'set', 'use', 'put', 'too', 'own',
    'how', 'who', 'now', 'man', 'way', 'day', 'try', 'run', 'hot',
    # Từ 2 ký tự phổ biến tiếng Anh
    'is', 'it', 'as', 'if', 'he', 'me', 'ok',
}

removed = []
for key in list(d.keys()):
    if key.lower() in english_common_short:
        canonical = d[key]
        print(f"  Remove ambiguous: {key!r} -> {canonical!r}")
        del d[key]
        removed.append(key)

print(f"\nRemoved {len(removed)} ambiguous English words")

# === Sửa "trung" -> "Trưng" (sai) ===
# "trung" (không có ư) ≠ "trưng" (có ư trong trưng bày/Hai Bà Trưng)
# Nếu base là "trung" thì canonical đúng phải là "Trung" (trung tâm, trung học)
# KHÔNG phải "Trưng" (vì strip_accents("Trưng") = "trung" nhưng cũng = "trung")
# → Vấn đề: strip_accents("Trưng") = "trung" (ư → u)
# Giải pháp: canonical "Trưng" chỉ đúng khi trong ngữ cảnh tên riêng (Hai Bà Trưng)
# Nhưng khi OCR đọc "trung" thì thường là "Trung" (trung tâm)
# → Đổi lại thành "Trung" an toàn hơn
if 'trung' in d:
    print(f"\nFix 'trung': {d['trung']!r} -> 'Trung'")
    d['trung'] = 'Trung'

# === Sửa "trong" -> "Trọng" (có thể sai) ===
# "trong" = "trong" (inside) KHÔNG PHẢI "Trọng" (tên riêng Trọng)
# OCR đọc "trong" thường là "trong" (vươn trong phố) → canonical đúng là "Trong"
# Nhưng "Trọng" có ọ khác base là "Trong" (strip: trong)
# strip_accents("Trọng") = "trong" - đúng
# Vấn đề: cả "Trong" và "Trọng" đều có base "trong"
# Từ OSM chọn "Trọng" vì xuất hiện nhiều (tên riêng Lê Văn Trọng, Trọng Tấn...)
# Nhưng trong OCR context "vườn trong phố" → "trong" = giới từ/tính từ, nên giữ nguyên
# GIẢI PHÁP: loại "trong" khỏi dict (quá ambiguous)
if 'trong' in d:
    print(f"\nRemove ambiguous 'trong': {d['trong']!r} (could be preposition)")
    del d['trong']

# === Kiểm tra thêm "trang" ===
if 'trang' in d:
    print(f"\n'trang': {d['trang']!r}")
    # "Trang" (trang trí, trang sức) OK nếu canonical là "Trang"
    # Nếu là "Tráng" thì sai (trang ≠ tráng về base: strip("Tráng")="trang" - đúng!)
    # strip("Tráng") = "trang" vì á → a? NO: á → a, ả → a, tất cả → a
    # Nhưng "trang" viết đúng là "trang", "tráng" (2 từ khác nhau)
    # → Loại bỏ vì quá ambiguous
    print(f"  Remove ambiguous 'trang'")
    del d['trang']

# === Loại thêm các từ rất ngắn (2 ký tự) có nguy cơ cao bị nhầm ===
short_risky = ['ai', 'am', 'an', 'ao', 'ap', 'au', 'ba', 'co', 'da', 'de', 'di',
               'du', 'gi', 'ha', 'ho', 'la', 'le', 'lo', 'ma', 'me', 'mi', 'mo',
               'mu', 'na', 'ne', 'ni', 'nu', 'oi', 'pa', 'pi', 'ra', 're', 'ro',
               'ru', 'sa', 'ta', 'te', 'ti', 'tu', 'ui', 'um', 'un', 'up', 'va',
               'vi', 'vo', 'vu', 'xa', 'xe', 'xi', 'xo', 'xu', 'ya']
for key in list(d.keys()):
    if key in short_risky:
        print(f"  Remove risky short: {key!r} -> {d[key]!r}")
        del d[key]

# === Loại bỏ thêm các từ tiếng Anh ngắn 4-5 ký tự ===
english_medium = {
    'club', 'shop', 'cafe', 'beer', 'wine', 'food', 'bank', 'mart', 'park',
    'city', 'town', 'hall', 'gate', 'road', 'lane', 'view', 'star', 'plus',
    'mini', 'mega', 'auto', 'moto', 'tech', 'media', 'hotel', 'house', 'store',
    'center', 'street', 'market', 'office', 'station', 'garden', 'tower',
    'plaza', 'court', 'place', 'point', 'world', 'group', 'team',
}
for key in list(d.keys()):
    if key.lower() in english_medium:
        print(f"  Remove English: {key!r} -> {d[key]!r}")
        del d[key]

print(f"\nAfter cleanup: {len(d)} entries")

# Save
with open('data/osm_words.json', 'w', encoding='utf-8') as f:
    json.dump(d, f, ensure_ascii=False, indent=2, sort_keys=True)

print("Saved osm_words.json")
