"""
Bổ sung manual entries vào osm_words.json cho các từ thường gặp trong OCR nhưng
OSM dataset không đủ sample.

Chiến lược: Thêm các từ phổ biến nhất trong tên POI tại TP.HCM mà OCR thường bỏ dấu
"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open('data/osm_words.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

print(f"Before: {len(d)} entries")

# === Manual entries: từ thường gặp trong tên POI tại TP.HCM ===
# Format: "base_không_dấu": "Dạng_chuẩn_có_dấu"
# Chỉ thêm từ phổ biến, thông dụng trong ngữ cảnh địa điểm/dịch vụ
manual_entries = {
    # === Dịch vụ thông dụng ===
    "nha hang": "Nhà hàng",
    "ca phe": "Cà phê",
    "quan ca phe": "Quán cà phê",
    "cua hang": "Cửa hàng",
    "tiem": "Tiệm",
    "khach san": "Khách sạn",
    "benh vien": "Bệnh viện",
    "bao tang": "Bảo tàng",
    "san bay": "Sân bay",
    "ben xe": "Bến xe",
    "ben tau": "Bến tàu",
    "giat ui": "Giặt ủi",
    "giat say": "Giặt sấy",
    "bai giu xe": "Bãi giữ xe",
    "bai do xe": "Bãi đỗ xe",
    "vuon hoa": "Vườn hoa",
    "nha sach": "Nhà sách",
    "phong kham": "Phòng khám",
    "nha khoa": "Nha khoa",
    "tham my vien": "Thẩm mỹ viện",
    "tham my": "Thẩm mỹ",
    "phong tap": "Phòng tập",
    "trung tam": "Trung tâm",
    "trung tam thuong mai": "Trung tâm thương mại",
    "tram xang": "Trạm xăng",
    "tram bang": "Trạm băng",
    "nha van hoa": "Nhà văn hóa",
    "thu vien": "Thư viện",
    "hoi truong": "Hội trường",
    "van phong": "Văn phòng",
    "cong ty": "Công ty",
    "chi nhanh": "Chi nhánh",
    "giao dich": "Giao dịch",
    "phong giao dich": "Phòng giao dịch",

    # === Ẩm thực ===
    "com tam": "Cơm tấm",
    "com binh dan": "Cơm bình dân",
    "pho bo": "Phở bò",
    "pho ga": "Phở gà",
    "bun bo": "Bún bò",
    "bun rieu": "Bún riêu",
    "hu tieu": "Hủ tiếu",
    "banh mi": "Bánh mì",
    "banh cuon": "Bánh cuốn",
    "banh xeo": "Bánh xèo",
    "lau": "Lẩu",
    "nuoc mia": "Nước mía",
    "tra sua": "Trà sữa",
    "bia hoi": "Bia hơi",

    # === Đường phố Quận 1 hay bị OCR sai ===
    "dong khoi": "Đồng Khởi",
    "le loi": "Lê Lợi",
    "hai ba trung": "Hai Bà Trưng",
    "dinh tien hoang": "Đinh Tiên Hoàng",
    "bao tang chung tich": "Bảo tàng Chứng tích",
    "duong sach": "Đường sách",
    "pho di bo": "Phố đi bộ",
    "cho ben thanh": "Chợ Bến Thành",
    "nha hat thanh pho": "Nhà hát Thành phố",
    "nha tho duc ba": "Nhà thờ Đức Bà",
    "dinh thong nhat": "Dinh Thống Nhất",
    "dinh doc lap": "Dinh Độc Lập",
    "buu dien trung tam": "Bưu điện Trung tâm",
    "ho con rua": "Hồ Con Rùa",
    "cong vien tao dan": "Công viên Tao Đàn",

    # === Từ đơn thường bị OCR sai dấu ===
    "duong": "Đường",
    "pho": "Phố",
    "ngo": "Ngõ",
    "hem": "Hẻm",
    "khu": "Khu",
    "toa": "Tòa",
    "tang": "Tầng",
    "phong": "Phòng",
    "so": "Số",
    "goc": "Góc",
    "dau": "Đầu",
    "cuoi": "Cuối",
    "giua": "Giữa",
    "truoc": "Trước",
    "sau": "Sau",
    "gan": "Gần",
    "canh": "Cạnh",
    "doi dien": "Đối diện",

    # === Tên viết tắt thường gặp ===
    "tp": "TP",
    "tp hcm": "TP.HCM",
    "q": "Q.",
    "p": "P.",
    "f": "P.",  # OCR hay đọc P thành F

    # === Tính từ và từ mô tả thường bị OCR bỏ dấu ===
    "dep": "Đẹp",
    "dep cafe": "Đẹp Café",
    "ngon": "Ngon",
    "sach": "Sạch",
    "sang": "Sang",
    "re": "Rẻ",
    "chat luong": "Chất lượng",
    "moi": "Mới",
    "xin": "Xịn",
    "vui": "Vui",
    "thuong": "Thương",
}

# Cập nhật: chỉ thêm nếu chưa có (ưu tiên OSM data)
added = 0
for base, canonical in manual_entries.items():
    if base not in d:
        d[base] = canonical
        added += 1
    # else: giữ nguyên OSM canonical (thường đúng hơn)

print(f"Added {added} manual entries")
print(f"After: {len(d)} entries")

# Lưu lại
with open('data/osm_words.json', 'w', encoding='utf-8') as f:
    json.dump(d, f, ensure_ascii=False, indent=2, sort_keys=True)

print("\nUpdated osm_words.json saved.")

# Kiểm tra lại
check = [
    'buu dien', 'duong sach', 'bai giu xe', 'vuon', 'nha hang',
    'ca phe', 'com tam', 'banh mi', 'pho bo', 'bun bo',
    'dong khoi', 'le loi', 'hai ba trung',
    'ngan hang', 'benh vien', 'cong vien', 'sieu thi',
]
print("\nVerification:")
for kw in check:
    val = d.get(kw)
    status = f"✓ → {val!r}" if val else "✗ not found"
    print(f"  {kw:25s} {status}")
