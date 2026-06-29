"""
build_osm_dict.py — Xây dựng từ điển tiếng Việt toàn quốc từ dữ liệu OSM Overpass API

Mục đích:
  - Trích xuất tên POI/đường/địa danh từ OSM tại nhiều trung tâm đô thị lớn ở Việt Nam
    (Hà Nội, TP.HCM, Đà Nẵng, Cần Thơ, Huế, Hải Phòng)
  - Đảm bảo từ điển bao phủ rộng rãi, phục vụ quét hàng nghìn địa điểm khác nhau trên cả nước
  - Áp dụng các quy tắc lọc bỏ từ mập mờ (ambiguous) và thêm các từ vá thủ công thông dụng
  - Xuất từ điển: base_không_dấu → dạng_có_dấu_phổ_biến_nhất

Cách dùng:
  python scripts/build_osm_dict.py
  → Tạo ra data/osm_words.json
"""

import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# === Cấu hình các trung tâm đô thị lớn tại Việt Nam ===
HUBS = [
    {"name": "TP.HCM", "lat": 10.779856, "lng": 106.699844},
    {"name": "Hà Nội", "lat": 21.028511, "lng": 105.804817},
    {"name": "Đà Nẵng", "lat": 16.054407, "lng": 108.202164},
    {"name": "Cần Thơ", "lat": 10.03711, "lng": 105.78825},
    {"name": "Huế", "lat": 16.463713, "lng": 107.590866},
    {"name": "Hải Phòng", "lat": 20.844911, "lng": 106.688087},
]

RADIUS_KM = 4.0  # Bán kính quét cho mỗi đô thị
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "osm_words.json"
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]

MAX_RETRIES = 3
RETRY_DELAY = 5  # giây

# === Regex patterns ===
_VIET_SHAPED = re.compile(
    r'[ăằắẳẵặâầấẩẫậêềếểễệôồốổỗộơờớởỡợưừứửữựđ]',
    re.IGNORECASE
)
_VIET_ACCENT = re.compile(
    r'[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ]'
)
_TOKEN_RE = re.compile(r'[A-Za-zÀ-ỹĐđ]+')

# === Accent stripping ===
_ACCENT_MAP = (
    (re.compile(r'[àáảãạăằắẳẵặâầấẩẫậ]'), 'a'),
    (re.compile(r'[èéẻẽẹêềếểễệ]'), 'e'),
    (re.compile(r'[ìíỉĩị]'), 'i'),
    (re.compile(r'[òóỏõọôồốổỗộơờớởỡợ]'), 'o'),
    (re.compile(r'[ùúủũụưừứửữự]'), 'u'),
    (re.compile(r'[ỳýỷỹỵ]'), 'y'),
    (re.compile(r'[đ]'), 'd'),
)


def strip_accents(text: str) -> str:
    """Bỏ dấu tiếng Việt, chuyển về chữ thường."""
    s = (text or '').lower()
    for pattern, repl in _ACCENT_MAP:
        s = pattern.sub(repl, s)
    return s


def has_vietnamese_mark(token: str) -> bool:
    """True nếu token có ký tự tiếng Việt đặc trưng."""
    return bool(_VIET_ACCENT.search(token or ''))


def query_overpass(lat: float, lng: float, radius_m: float) -> dict:
    """
    Truy vấn Overpass API lấy tên OSM trong bán kính.
    Lấy: nodes, ways, relations có tag name/name:vi/alt_name.
    """
    query = (
        f'[out:json][timeout:60];'
        f'('
        f'node["name"](around:{int(radius_m)},{lat},{lng});'
        f'way["name"](around:{int(radius_m)},{lat},{lng});'
        f'relation["name"](around:{int(radius_m)},{lat},{lng});'
        f'node["name:vi"](around:{int(radius_m)},{lat},{lng});'
        f'way["name:vi"](around:{int(radius_m)},{lat},{lng});'
        f');'
        f'out tags;'
    )
    post_body = urllib.parse.urlencode({'data': query}).encode('utf-8')

    for server_url in OVERPASS_URLS:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"  Querying {server_url} (attempt {attempt})...")
                req = urllib.request.Request(
                    server_url,
                    data=post_body,
                    headers={
                        'User-Agent': 'OSM-Nationwide-Dict-Builder/2.0',
                        'Content-Type': 'application/x-www-form-urlencoded',
                    }
                )
                with urllib.request.urlopen(req, timeout=90) as resp:
                    raw = resp.read().decode('utf-8')
                result = json.loads(raw)
                return result
            except urllib.error.HTTPError as e:
                body = ''
                try:
                    body = e.read().decode('utf-8', errors='replace')[:200]
                except Exception:
                    pass
                print(f"  HTTP {e.code} from {server_url}: {e.reason} — {body}")
                if e.code == 429:
                    time.sleep(RETRY_DELAY * attempt * 2)
                elif attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                else:
                    break
            except Exception as exc:
                print(f"  Error from {server_url}: {exc}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                else:
                    break

    return {}


def extract_names_from_osm(osm_data: dict) -> List[str]:
    """Trích xuất tất cả tên từ kết quả Overpass."""
    names = []
    name_tags = ('name', 'name:vi', 'alt_name', 'official_name', 'short_name')

    for element in osm_data.get('elements', []):
        tags = element.get('tags', {})
        for tag in name_tags:
            val = tags.get(tag, '').strip()
            if val:
                names.append(val)

    return names


def build_word_frequency(names: List[str]) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
    """Xây dựng bảng tần suất từ đơn và bigram."""
    word_freq = defaultdict(lambda: defaultdict(int))
    bigram_freq = defaultdict(lambda: defaultdict(int))

    for name in names:
        tokens = _TOKEN_RE.findall(name)
        viet_tokens = []

        for tok in tokens:
            if not has_vietnamese_mark(tok):
                continue
            if tok.isupper() and len(tok) > 3:
                continue
            if any(ch.isdigit() for ch in tok):
                continue

            base = strip_accents(tok)
            if len(base) < 2:
                continue

            word_freq[base][tok] += 1
            viet_tokens.append((base, tok))

        # Bigrams
        for i in range(len(viet_tokens) - 1):
            b1, t1 = viet_tokens[i]
            b2, t2 = viet_tokens[i + 1]
            bigram_base = f"{b1} {b2}"
            bigram_text = f"{t1} {t2}"
            bigram_freq[bigram_base][bigram_text] += 1

    return word_freq, bigram_freq


def select_canonical(freq_map: Dict[str, int], min_count: int = 1) -> Tuple[str, int]:
    """Chọn dạng chuẩn có dấu tối ưu từ tập tần suất."""
    if not freq_map:
        return None, 0

    filtered = {k: v for k, v in freq_map.items() if v >= min_count}
    if not filtered:
        filtered = freq_map

    def score(item):
        form, count = item
        viet_score = len(_VIET_SHAPED.findall(form))
        title_bonus = 1 if form[0].isupper() else 0
        return (count, viet_score, title_bonus)

    best_form, best_count = max(filtered.items(), key=score)
    return best_form, best_count


def build_dictionary(
    word_freq: Dict[str, Dict[str, int]],
    bigram_freq: Dict[str, Dict[str, int]],
    min_word_count: int = 2,
    min_bigram_count: int = 3,
) -> dict:
    """Xây dựng từ điển thô trước khi áp dụng các bộ lọc."""
    dictionary = {}

    # Từ đơn
    for base, freq_map in word_freq.items():
        canonical, count = select_canonical(freq_map, min_count=min_word_count)
        if canonical and count >= min_word_count:
            dictionary[base] = canonical

    # Bigrams
    for base, freq_map in bigram_freq.items():
        canonical, count = select_canonical(freq_map, min_count=min_bigram_count)
        if canonical and count >= min_bigram_count:
            parts = base.split()
            if len(parts) == 2:
                w1_canon = dictionary.get(parts[0])
                w2_canon = dictionary.get(parts[1])
                if w1_canon and w2_canon:
                    expected = f"{w1_canon} {w2_canon}"
                    if canonical != expected:
                        dictionary[base] = canonical
                else:
                    dictionary[base] = canonical

    return dictionary


def clean_and_patch_dict(dictionary: dict) -> dict:
    """
    Áp dụng quy trình lọc bỏ các từ đơn ambiguous nguy hiểm
    và thêm các từ vá thủ công thông dụng.
    """
    print("\n--- Cleaning and Patching Dictionary ---")
    
    # 1. Các từ ngắn tiếng Anh hoặc từ viết tắt/tên riêng phổ biến
    english_common_short = {
        'so', 'no', 'to', 'the', 'in', 'on', 'at', 'of', 'or', 'an',
        'be', 'do', 'go', 'up', 'us', 'by', 'my', 'we',
        'not', 'but', 'for', 'are', 'was', 'has', 'had', 'its', 'our',
        'out', 'all', 'one', 'can', 'may', 'him', 'his', 'her',
        'any', 'old', 'new', 'big', 'set', 'use', 'put', 'too', 'own',
        'how', 'who', 'now', 'man', 'way', 'day', 'try', 'run', 'hot',
        'is', 'it', 'as', 'if', 'he', 'me', 'ok',
    }

    # 2. Các từ tiếng Việt có nhiều nghĩa dấu khác nhau (ambiguous) và cực kỳ phổ biến
    harmful_words = {
        # Quốc gia/địa danh phổ biến
        'viet', 'nga', 'nam', 'han', 'nhat', 'phap', 'my', 'anh', 'thai', 'lao', 'trung', 'an', 'philippines', 'singapore', 'malaysia',
        # Hướng/vị trí
        'dong', 'tay', 'nam', 'bac', 'trung', 'giua', 'trong', 'ngoai', 'tren', 'duoi',
        # Đơn vị hành chính/dịch vụ
        'quan', 'phuong', 'tinh', 'huyen', 'xa', 'thanh', 'pho', 'quoc', 'gia', 'dinh',
        # Từ vựng thông dụng dễ bị sai dấu lệch
        'ngon', 'mai', 'mon', 'moc', 'hoa', 'hai', 'ba', 'bon', 'nam', 'sau', 'bay', 'tam', 'chin',
        # Tên người/tên riêng
        'minh', 'duc', 'tuan', 'linh', 'son', 'van', 'phong', 'vy', 'khoa', 'huy',
        'hoang', 'khanh', 'trang', 'dung', 'anh', 'yen', 'oanh', 'nguyet', 'hang',
        'lan', 'mai', 'cuc', 'truc', 'dao', 'le', 'lieu', 'hong', 'que',
        # Từ viết tắt hành chính/địa lý
        'tp', 'hcm', 'hn', 'dn', 'hp', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9',
        # Giới từ/đại từ
        'cua', 'cho', 'nhung', 'cac', 'mot', 'nhieu', 'it', 'va', 'co', 'khong', 'la', 'di', 'den', 've', 'ra', 'vao', 'len', 'xuong',
        # Các từ Hán-Việt ambiguous
        'tinh', 'binh', 'hung', 'thi', 'phat', 'long', 'tuan', 'cuong', 'sang', 'nhan', 'bach', 'linh', 'dat', 'loi', 'tung'
    }

    # 3. Lọc bỏ các từ đơn (không chứa khoảng trắng) có độ dài <= 3 ký tự
    # để bảo vệ các brand, từ viết tắt, tiếng Anh ngắn (ngoại trừ whitelist tiếng Việt phổ biến)
    viet_short_whitelist = {
        "nha", "nam", "sam", "cho", "pho", "ngo", "hem", "khu", "toa", "tang", "sau", "gan", "canh", "ben", 
        "cau", "den", "rap", "ban", "am", "ho", "y", "ruo", "tiem", "lau", "com", "bun", "tra", "sua", "bia", 
        "hoi", "xe", "may", "sua", "tuy", "yen", "cuc", "dao", "mai", "lan", "truc", "le", "hong", "que",
        "tuan", "dat", "loi", "tung", "nhi", "nga", "hai", "ba", "bon", "sau", "bay", "tam", "chi", "gia", 
        "ong", "anh", "em", "co", "di", "mo", "bo", "me", "con", "cua", "ga", "bo", "heo", "de", "ca", "oc", 
        "nuo", "can", "mi",
    }
    
    removed_count = 0
    for key in list(dictionary.keys()):
        key_lower = key.lower()
        if key_lower in viet_short_whitelist:
            continue
        if (
            key_lower in english_common_short
            or key_lower in harmful_words
            or (' ' not in key and len(key) <= 3)
        ):
            del dictionary[key]
            removed_count += 1

    print(f"  → Filtered out {removed_count} ambiguous or short single-word entries.")

    # 4. Bổ sung manual patches cho các cụm từ phổ biến (không bị đè bởi quy tắc lọc)
    manual_entries = {
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
        "nha van hoa": "Nhà văn hóa",
        "thu vien": "Thư viện",
        "hoi truong": "Hội trường",
        "van phong": "Văn phòng",
        "cong ty": "Công ty",
        "chi nhanh": "Chi nhánh",
        "giao dich": "Giao dịch",
        "phong giao dich": "Phòng giao dịch",
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
        "dong khoi": "Đồng Khởi",
        "le loi": "Lê Lợi",
        "hai ba trung": "Hai Bà Trưng",
        "dinh tien hoang": "Đinh Tiên Hoàng",
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
        "duong": "Đường",
        "pho": "Phố",
        "ngo": "Ngõ",
        "hem": "Hẻm",
        "khu": "Khu",
        "toa": "Tòa",
        "tang": "Tầng",
        "phong": "Phòng",
        "goc": "Góc",
        "dau": "Đầu",
        "cuoi": "Cuối",
        "giua": "Giữa",
        "truoc": "Trước",
        "sau": "Sau",
        "gan": "Gần",
        "canh": "Cạnh",
        "doi dien": "Đối diện",
        "tp hcm": "TP.HCM",
    }

    added = 0
    for base, canonical in manual_entries.items():
        if base not in dictionary:
            dictionary[base] = canonical
            added += 1
            
    print(f"  → Added {added} manual service and place tokens.")
    return dictionary


def main():
    print("=== OSM Nationwide Dictionary Builder ===")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Aggregating data from {len(HUBS)} major cities in Vietnam...")
    print()

    radius_m = RADIUS_KM * 1000
    all_names = []

    # 1. Query Overpass API for each hub
    for hub in HUBS:
        print(f"\nQuerying hub: {hub['name']} ({hub['lat']}, {hub['lng']})")
        try:
            osm_data = query_overpass(hub['lat'], hub['lng'], radius_m)
            elements = osm_data.get('elements', [])
            print(f"  → Got {len(elements)} OSM elements")
            
            names = extract_names_from_osm(osm_data)
            print(f"  → Extracted {len(names)} name strings")
            all_names.extend(names)
        except Exception as exc:
            print(f"  → Failed to query {hub['name']}: {exc}")

    # Nếu hoàn toàn không lấy được dữ liệu, dùng cache nếu có
    if not all_names:
        print("\nAll Overpass queries failed. Trying local cache...")
        cache_path = OUTPUT_PATH.parent / "osm_raw_cache_nationwide.json"
        if cache_path.exists():
            with open(cache_path, 'r', encoding='utf-8') as f:
                all_names = json.load(f)
            print(f"  → Loaded {len(all_names)} name strings from cache")
        else:
            print("  No cache available. Exiting.")
            sys.exit(1)
    else:
        # Lưu cache để dùng lần sau
        cache_path = OUTPUT_PATH.parent / "osm_raw_cache_nationwide.json"
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(all_names, f, ensure_ascii=False, indent=2)
            print(f"\n  → Cached raw names to {cache_path.name}")
        except Exception as cache_err:
            print(f"  → Failed to write cache: {cache_err}")

    # 2. Build frequency tables
    print("\nBuilding nationwide word frequency table...")
    word_freq, bigram_freq = build_word_frequency(all_names)
    print(f"  → {len(word_freq)} unique base words")
    print(f"  → {len(bigram_freq)} unique base bigrams")

    # 3. Build dictionary
    print("\nBuilding canonical dictionary...")
    dictionary = build_dictionary(word_freq, bigram_freq, min_word_count=2, min_bigram_count=3)
    
    # 4. Clean and patch
    dictionary = clean_and_patch_dict(dictionary)
    print(f"  → Final dictionary has {len(dictionary)} entries.")

    # 5. Save dictionary
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"\n✓ Dictionary saved to {OUTPUT_PATH}")

    # 6. Validation
    print("\n=== Validation: Key words ===")
    key_words = [
        'buu dien', 'duong sach', 'bai giu xe', 'vuon', 'nha hang',
        'quan ca phe', 'com tam', 'nguyen hue', 'hai ba trung',
        'cong vien', 'bao tang', 'ngan hang', 'truong', 'benh vien',
        'chua', 'nha tho', 'sieu thi', 'phong kham', 'khach san',
        'trung tam', 'cua hang', 'tram xang', 'banh mi', 'pho bo'
    ]
    found = 0
    for kw in key_words:
        if kw in dictionary:
            print(f"  ✓ {kw!r:25s} → {dictionary[kw]!r}")
            found += 1
        else:
            print(f"  ✗ {kw!r:25s} not found")
    print(f"\n  Found {found}/{len(key_words)} key words ({100*found//len(key_words)}%)")


if __name__ == '__main__':
    main()
