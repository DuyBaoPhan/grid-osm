"""
Kiểm tra toàn bộ từ điển osm_words.json và lọc bỏ các từ đơn (single tokens)
có nguy cơ gây sai lệch (over-correction) cao.
"""
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('data/osm_words.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

print(f"Original dictionary size: {len(d)}")

# === Danh sách các từ cần loại bỏ khỏi từ điển đơn ===
# 1. Các từ tiếng Anh hoặc từ viết tắt/tên riêng phổ biến
# 2. Các từ tiếng Việt có nhiều nghĩa dấu khác nhau (ambiguous) và cực kỳ phổ biến
harmful_words = {
    # Tên riêng quốc gia/địa danh phổ biến
    'viet', 'nga', 'nam', 'han', 'nhat', 'phap', 'my', 'anh', 'thai', 'lao', 'trung', 'an', 'philippines', 'singapore', 'malaysia',
    
    # Từ chỉ hướng/vị trí (thường dùng làm tên đường, tên riêng hoặc từ thường)
    'dong', 'tay', 'nam', 'bac', 'trung', 'giua', 'trong', 'ngoai', 'tren', 'duoi',
    
    # Từ chỉ chức vụ/nghề nghiệp/đơn vị hành chính dễ nhầm
    'quan', 'phuong', 'tinh', 'huyen', 'xa', 'thanh', 'pho', 'quoc', 'gia', 'dinh',
    
    # Từ mô tả tính chất/từ vựng ẩm thực/tên riêng dễ bị đổi dấu sai lệch
    'ngon',  # Ngon (ngon miệng) vs Ngôn (tên)
    'mai',   # Mai (hoa mai, ngày mai) vs Mại (khuyến mại) vs Mái
    'mon',   # Món (món ăn) vs Mòn
    'moc',   # Mộc (gỗ, mộc mạc) vs Mốc (nấm mốc)
    'hoa',   # Hoa (bông hoa, tên riêng) vs Hóa (hóa học) vs Họa
    'hai',   # Hải (biển) vs Hài (hài hước) vs Hai (số 2)
    'ba',    # Ba (số 3, ba mẹ) vs Bà (bà nội)
    'bon',   # Bốn vs Bon
    'nam',   # Nam (phương nam, nam giới) vs Năm (số 5, năm học)
    'sau',   # Sáu vs Sau (phía sau) vs Sâu
    'bay',   # Bảy vs Bay (bay lượn) vs Bày
    'tam',   # Tám vs Tâm vs Tấm
    'chin',  # Chín vs Chiên
    
    # Tên người/tên riêng cực kỳ phổ biến
    'minh', 'duc', 'tuan', 'linh', 'son', 'van', 'phong', 'vy', 'khoa', 'huy',
    'hoang', 'khanh', 'trang', 'dung', 'anh', 'yen', 'oanh', 'nguyet', 'hang',
    'lan', 'mai', 'cuc', 'truc', 'dao', 'le', 'lieu', 'hong', 'que',
    
    # Từ viết tắt
    'tp', 'hcm', 'hn', 'dn', 'hp', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9',
    
    # Các giới từ/đại từ/từ nối khác
    'cua',   # Của (sở hữu) vs Cửa (cửa ra vào) vs Cưa
    'cho',   # Chợ vs Cho (tặng)
    'nhung', # Những vs Nhung (vải)
    'cac',   # Các vs Các (cát)
    'mot',   # Một vs Mọt
    'nhieu', # Nhiều vs Nhiêu
    'it',    # Ít vs It (tiếng Anh)
    'va',    # Và vs Vá
    'co',    # Có vs Cô vs Cọ
    'khong', # Không vs Khống
    'la',    # Là vs Lá vs Lạ
    'di',    # Đi vs Dì
    'den',   # Đến vs Đèn vs Đen
    've',    # Về vs Vẽ vs Vé
    'ra',    # Ra vs Rạ
    'vao',   # Vào vs Váo
    'len',   # Lên vs Len (vải len)
    'xuong', # Xuống vs Xuổng
}

viet_short_whitelist = {
    # Loại hình / Loại địa điểm / Động từ / Danh từ phổ biến
    "nha", "nam", "sam", "cho", "pho", "ngo", "hem", "khu", "toa", "tang", "sau", "gan", "canh", "ben", 
    "cau", "den", "rap", "ban", "am", "ho", "y", "ruo", "tiem", "lau", "com", "bun", "tra", "sua", "bia", 
    "hoi", "xe", "may", "sua", "tuy", "yen", "cuc", "dao", "mai", "lan", "truc", "le", "hong", "que",
    "tuan", "dat", "loi", "tung", "nhi", "nga", "hai", "ba", "bon", "sau", "bay", "tam", "chi", "gia", 
    "ong", "anh", "em", "co", "di", "mo", "bo", "me", "con", "cua", "ga", "bo", "heo", "de", "ca", "oc", 
    "nuo", "can", "mi",
}

removed_count = 0
for w in harmful_words:
    if w.lower() in viet_short_whitelist:
        continue
    if w in d:
        print(f"Removing harmful entry: {w!r} -> {d[w]!r}")
        del d[w]
        removed_count += 1

for k in list(d.keys()):
    if ' ' not in k and len(k) <= 3:
        if k.lower() in viet_short_whitelist:
            continue
        # Nếu từ này có độ dài <= 3 và không chứa khoảng trắng, loại bỏ để an toàn
        # (Ngoại lệ: chỉ giữ các từ cực kỳ chắc chắn nếu cần, nhưng tốt nhất loại bỏ hết từ ngắn đơn lẻ)
        val = d[k]
        print(f"Removing short single-word entry: {k!r} -> {val!r}")
        del d[k]
        removed_count += 1

print(f"\nTotal removed: {removed_count} entries")
print(f"Final dictionary size: {len(d)} entries")

# Ghi lại osm_words.json
with open('data/osm_words.json', 'w', encoding='utf-8') as f:
    json.dump(d, f, ensure_ascii=False, indent=2, sort_keys=True)

print("osm_words.json updated successfully.")
