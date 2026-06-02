# =============================================================
# vision.py — Gọi Ollama Local LLM (Qwen2.5-VL)
#
# Nhận screenshot bytes → trả về list tên POI và cờ báo ngoài quận.
# Có retry logic và JSON fallback.
# =============================================================

import asyncio
import base64
import json
import logging
import re
from typing import List, Tuple

from openai import AsyncOpenAI

from config import OLLAMA_API_BASE, OLLAMA_MODEL, MAX_RETRIES, TARGET_DISTRICT

logger = logging.getLogger(__name__)

# ── Khởi tạo OpenAI-compat client cho Ollama ─────────────────
_client = AsyncOpenAI(
    base_url=OLLAMA_API_BASE,
    api_key="ollama",           # Ollama không cần API key thật
)

# ── System message (vai trò) ─────────────────────────────────
SYSTEM_MESSAGE = (
    "You are a professional OCR tool specialized in reading place labels from map images. "
    "Task: extract only PROPER NAMES of real places (shops, cafes, banks, post offices, hotels, landmarks, etc.). "
    "STRICT RULE: NEVER extract road/street names — they appear as small italic text running along road lines. "
    "NEVER invent or hallucinate any name. If you cannot read a label clearly, skip it."
)

# ── Prompt ───────────────────────────────────────────────────
_PROMPT_TEMPLATE = """\
You are an AI specialized in reading place labels and finding their precise icon positions from map images.
The image below is a clean map screenshot centered exactly at a known GPS coordinate.

=== MAP TILE INFO ===
Tile: ({tx}, {ty}) @ Zoom {Z}
Center GPS: {center_lat:.6f}°N, {center_lng:.6f}°E

=== IMAGE RESOLUTION & COORDINATES ===
- Width: {img_w} pixels
- Height: {img_h} pixels
- Center: ({center_x:.1f}, {center_y:.1f}) (This corresponds exactly to the Center GPS {center_lat:.6f}°N, {center_lng:.6f}°E)
- Top-left corner: (0, 0)
- Bottom-right corner: ({img_w}, {img_h})

=== HOW TO DISTINGUISH POIs FROM ROAD NAMES ===
✅ POI labels to EXTRACT (place names):
  - Colored text labels with a map icon/symbol (blue, brown, orange, red...)
  - Key: the label has a dedicated icon (building, shop, bank symbol, post office symbol, food/beverage cup etc.)
  - Examples: "Highlands Coffee", "Bưu điện Trung tâm Sài Gòn", "OCB", "Circle K"

❌ Road/street names — DO NOT extract:
  - Small italic/gray text running along road lines
  - Examples: "Lê Duẩn", "Nguyễn Văn Bình", "Hai Bà Trưng", "Đồng Khởi"

=== COORDINATE GUIDE ===
To locate the colored icon center precisely on this {img_w}x{img_h} image:
- Locate the exact center of the colored graphic icon/marker itself (the graphic symbol, e.g. the green envelope icon, coffee cup, bank symbol etc.).
- Use the actual physical pixels of the image as the coordinates (0 to {img_w} for x, and 0 to {img_h} for y).
- If the icon is physically above the center of the image, y MUST be less than {center_y:.1f}.
- If the icon is physically to the left of the center, x MUST be less than {center_x:.1f}.
- Estimate x and y based on the actual visual position of the icon graphic in physical pixels.

=== TASK ===
Scan the clean map image. For every place label CLEARLY VISIBLE on the map:
1. Read the EXACT name as printed (keep diacritics: ả,ầ,ị,ỏ,ư,đ... e.g. "Bánh Mì" not "Banh Mi")
2. Identify the precise pixel coordinate (x, y) in physical pixels of the POI's icon/symbol.
   DO NOT locate the text label. Locate the exact center of the colored graphic icon/marker itself.
   If the graphic icon/marker is missing (extremely rare), use the center of its text label.
3. First, write a one-sentence "reasoning" analyzing the spatial position of the icon relative to the center ({center_x:.1f}, {center_y:.1f}).
4. Then output the precise "x" and "y" coordinates in physical pixels.
5. CRITICAL: NEVER copy the example values ({ex_x:.1f}, {ex_y:.1f}) unless the icon is visually located at that exact location. Always estimate the real coordinates (x, y) from the visual position of the icon.

Return ONLY a JSON array:
[
  {{
    "name": "exact name from map",
    "type": "restaurant|cafe|bank|hotel|shop|hospital|school|post_office|parking|monument|church|park|square|landmark|other",
    "reasoning": "The icon is located in the {ex_tb} and to the {ex_lr} of the center ({center_x:.1f}, {center_y:.1f}), so its estimated position is x={ex_x:.1f}, y={ex_y:.1f}.",
    "x": {ex_x:.1f},
    "y": {ex_y:.1f},
    "confidence": 0.95
  }}
]
No POIs visible → return: []
"""


# ── Hàm chính ────────────────────────────────────────────────

async def extract_pois_from_screenshot(
    screenshot_bytes: bytes,
    bbox: Tuple[float, float, float, float] = None,
    tx: int = None,
    ty: int = None,
    zoom: int = None,
    img_metadata: dict = None,
) -> Tuple[List[dict], bool]:
    """
    Gửi screenshot → Ollama → trả về tuple (list dict tên POI + vị trí, cờ outside_district).
    Tự động retry `MAX_RETRIES` lần nếu lỗi.
    """
    img_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
    
    # Định dạng prompt với tọa độ thực địa
    if bbox:
        lat_min, lng_min, lat_max, lng_max = bbox
    else:
        import config as _cfg
        lat_min, lng_min = _cfg.CENTER_LAT - 0.001, _cfg.CENTER_LNG - 0.001
        lat_max, lng_max = _cfg.CENTER_LAT + 0.001, _cfg.CENTER_LNG + 0.001

    # Tính toán tọa độ tâm và trung điểm
    center_lat = (lat_min + lat_max) / 2.0
    center_lng = (lng_min + lng_max) / 2.0
    
    import config as _cfg
    z_val = zoom if zoom is not None else getattr(_cfg, "SCREENSHOT_ZOOM", 19)
    tx_val = tx if tx is not None else 0
    ty_val = ty if ty is not None else 0

    if img_metadata is None:
        img_metadata = {
            "width": 612,
            "height": 612,
            "center_x": 306.0,
            "center_y": 306.0,
            "scale": 2.0,
        }

    img_w = img_metadata.get("width", 612)
    img_h = img_metadata.get("height", 612)
    center_x = img_metadata.get("center_x", img_w / 2.0)
    center_y = img_metadata.get("center_y", img_h / 2.0)

    # Sinh tọa độ ví dụ ngẫu nhiên động để chặn đứng model học vẹt ví dụ tĩnh
    import random
    ex_x = round(random.uniform(img_w * 0.25, img_w * 0.75), 1)
    ex_y = round(random.uniform(img_h * 0.25, img_h * 0.75), 1)
    lr_desc = "left" if ex_x < center_x else "right"
    tb_desc = "upper area, clearly above" if ex_y < center_y else "lower area, clearly below"

    prompt = _PROMPT_TEMPLATE.format(
        tx=tx_val,
        ty=ty_val,
        Z=z_val,
        center_lat=center_lat,
        center_lng=center_lng,
        img_w=img_w,
        img_h=img_h,
        center_x=center_x,
        center_y=center_y,
        ex_x=ex_x,
        ex_y=ex_y,
        ex_lr=lr_desc,
        ex_tb=tb_desc,
    )

    for attempt in range(1, MAX_RETRIES + 2):  # 1..MAX_RETRIES+1
        try:
            response = await _client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_MESSAGE,
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_b64}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=800,
                temperature=0.0,
            )

            raw_text = response.choices[0].message.content or ""
            # Ghi lại log phản hồi thô từ LLM để hỗ trợ debug OCR trực tiếp trên console
            logger.info("  [Vision LLM Raw Response]:\n%s", raw_text)
            
            pois, outside = _parse_poi_response(raw_text)
            
            # Programmatic case-insensitive deduplication of dicts, preserving order
            seen_pois = set()
            unique_pois = []
            for p in pois:
                name_clean = str(p.get("name", "")).strip()
                name_lower = name_clean.lower()
                if name_clean and name_lower not in seen_pois:
                    seen_pois.add(name_lower)
                    poi_entry = {
                        "name": name_clean,
                    }
                    if "x" in p:
                        poi_entry["x"] = p["x"]
                    if "y" in p:
                        poi_entry["y"] = p["y"]
                    unique_pois.append(poi_entry)

            # Lọc bỏ các tên nghi là tên đường hoặc hallucination điển hình
            filtered_pois = []
            for p in unique_pois:
                if _is_hallucinated_name(p["name"]):
                    logger.warning(
                        "  [AntiHalluc] Bỏ tên nghi hallucination: '%s'", p["name"]
                    )
                elif _is_street_name(p["name"]):
                    logger.warning(
                        "  [AntiStreet] Bỏ tên đường phố: '%s'", p["name"]
                    )
                else:
                    filtered_pois.append(p)

            logger.debug("Vision OK: %d POIs extracted, outside=%s", len(filtered_pois), outside)
            return filtered_pois, outside

        except Exception as exc:
            if attempt <= MAX_RETRIES:
                wait = attempt * 2.0
                logger.warning(
                    "Vision attempt %d/%d failed (%s). Retry in %.1fs…",
                    attempt, MAX_RETRIES, exc, wait,
                )
                await asyncio.sleep(wait)
            else:
                logger.error("Vision failed after %d attempts: %s", MAX_RETRIES + 1, exc)
                return [], False

    return [], False  # unreachable but satisfies type checker


# ── Bộ lọc tên hallucination ─────────────────────────────────

# Pattern: "Cafe A", "Store B", "Restaurant C", "Coffee Shop D" v.v.
# LLM thường bịa tên dạng [danh từ thể loại] + [chữ cái A-Z đơn lẻ hoặc số]
_HALLUC_PATTERN = re.compile(
    r"""^(
        cafe|coffee|shop|store|restaurant|hotel|bank|market|church|
        school|park|pharmacy|clinic|spa|gym|bar|pub|hostel|supermarket|
        mall|center|centre|tower|building|office|station
    )\s+[a-z\d]$""",
    re.IGNORECASE | re.VERBOSE,
)
# Cũng lọc pattern "[Tên] [A-Z]" tức là một từ + một chữ cái đơn lẻ ở cuối
_SINGLE_LETTER_SUFFIX = re.compile(r'\s+[A-Z]$')

# Tập hợp các danh từ chỉ thể loại POI chung chung để lọc bỏ triệt để các hallucination một từ
_GENERIC_POI_NAMES = {
    "cafe", "coffee", "shop", "store", "restaurant", "hotel", "bank", "market",
    "church", "school", "park", "pharmacy", "clinic", "spa", "gym", "bar",
    "pub", "hostel", "supermarket", "mall", "center", "centre", "tower",
    "building", "office", "station", "post office", "post_office", "landmark",
    "nhà hàng", "quán ăn", "cà phê", "ngân hàng", "khách sạn", "trường học",
    "bệnh viện", "chợ", "công viên", "nhà thờ", "siêu thị", "tòa nhà", "văn phòng",
    "bưu điện", "trụ sở", "cửa hàng", "cửa hiệu", "hiệu thuốc", "quầy thuốc"
}

def _is_hallucinated_name(name: str) -> bool:
    """Trả về True nếu tên trông như hallucination điển hình của LLM hoặc là tên thể loại chung chung."""
    name = name.strip()
    name_lower = name.lower()
    
    # 1. Lọc bỏ các từ thể loại chung chung một từ đơn độc
    if name_lower in _GENERIC_POI_NAMES:
        return True
        
    # 2. Lọc các pattern hallucinated (dạng danh từ loại + chữ cái đơn)
    if _HALLUC_PATTERN.match(name):
        return True
        
    # 3. Lọc pattern "[Từ bất kỳ] [Chữ cái đơn]"
    if _SINGLE_LETTER_SUFFIX.search(name) and len(name.split()) <= 3:
        return True
        
    return False


# ── Bộ lọc tên đường giao thông (Universal Street Name Filter) ───
# Không dùng danh sách cứng theo quận — nhận dạng bằng cấu trúc pattern
# để hoạt động linh hoạt với mọi quận/tỉnh/thành phố.

# ① Từ khóa giao thông rõ ràng (tiền tố/loại đường)
_STREET_KEYWORDS = re.compile(
    r"\b(đường|phố|đại lộ|quốc lộ|tỉnh lộ|liên tỉnh|liên huyện"
    r"|boulevard|avenue|street|road|lane|alley|st\b|rd\b|ave\b|ln\b|blvd\b"
    r"|hẻm|ngõ|kiệt|xóm|thôn\s+\d"
    r"|vòng xoay|nút giao|ngã tư|ngã năm|ngã sáu|ngã bảy|ngã ba"
    r"|cầu\s+\w|cống\s+\w"
    r")\b",
    re.IGNORECASE,
)

# ② Cụm cần loại bỏ trước khi kiểm tra (tránh false positive)
_FALSE_POSITIVE_PHRASES = re.compile(
    r"\b(thành\s*phố|thành\s*thị|thị\s*xã|thị\s*trấn|tỉnh\s+\w+|quận\s+\d+|huyện\s+\w+)\b",
    re.IGNORECASE,
)

# ③ Pattern nhận dạng cấu trúc tên đường Việt Nam KHÔNG có từ khóa tiền tố:
#    - "Số [N]" hoặc "[N]/[M]" — đường có số hiệu (QL1, TL13, D1, D2...)
_NUMBERED_ROAD = re.compile(
    r"^(ql|tl|đt|nh|d|n|b|c|r|f|g|h|k|p|q|s|t|u|v|w|x|y|z)\s*\d+[a-z]?$",
    re.IGNORECASE,
)


def _strip_vietnamese_accents(s: str) -> str:
    """Loại bỏ hoàn toàn dấu tiếng Việt và đưa về chữ thường."""
    s = s.lower()
    s = re.sub(r"[àáảãạăằắẳẵặâầấẩẫậ]", "a", s)
    s = re.sub(r"[èéẻẽẹêềếểễệ]", "e", s)
    s = re.sub(r"[ìíỉĩị]", "i", s)
    s = re.sub(r"[òóỏõọôồốổỗộơờớởỡợ]", "o", s)
    s = re.sub(r"[ùúủũụưừứửữự]", "u", s)
    s = re.sub(r"[ỳýỷỹỵ]", "y", s)
    s = re.sub(r"[đ]", "d", s)
    return s


def _is_street_name(name: str) -> bool:
    """
    Nhận dạng tên đường giao thông bằng pattern linh hoạt — hoạt động với
    mọi quận/tỉnh/thành phố, không cần hardcode danh sách đường.

    Logic nhận dạng:
      1. Loại bỏ false-positive phrases (thành phố, quận, huyện...) khỏi chuỗi
         trước khi kiểm tra, tránh nhầm "Bưu điện Thành Phố" là tên đường.
      2. Kiểm tra từ khóa tiền tố giao thông rõ ràng (đường, phố, hẻm...).
      3. Kiểm tra tiền tố số-hiệu đường (QL1, D3, TL13...).
      4. Kiểm tra cấu trúc "[Tiền tố người Việt] + [Họ tên]" đứng độc lập
         (không có icon/loại hình đi kèm) — tên đường Việt Nam thường dùng
         họ tên anh hùng dân tộc 2-4 từ, không có từ chỉ loại hình.
    """
    name_clean = name.strip()
    if not name_clean:
        return False
    name_lower = name_clean.lower()

    # Bước 1: bóc tách false-positive trước khi kiểm tra keyword
    check_str = _FALSE_POSITIVE_PHRASES.sub("", name_lower).strip()

    # Bước 2: từ khóa giao thông rõ ràng
    if _STREET_KEYWORDS.search(check_str):
        return True

    # Bước 3: tên đường có số hiệu (D1, QL13, TL9B...)
    no_accent = _strip_vietnamese_accents(check_str)
    if _NUMBERED_ROAD.match(no_accent.strip()):
        return True

    # Bước 4: Heuristic cấu trúc tên đường Việt Nam
    # Tên đường VN thường là: [họ] [đệm] [tên] — 2 đến 5 từ đơn, không có
    # từ nào chỉ loại hình dịch vụ (plaza, coffee, bank, hotel...).
    # Chỉ áp dụng khi chuỗi hoàn toàn là chữ (không có số, dấu ngoặc, ký hiệu).
    words = check_str.split()
    if 2 <= len(words) <= 5 and re.match(r"^[a-z\s]+$", no_accent):
        # Kiểm tra không phải tên POI thực sự bằng cách loại trừ
        # các từ chỉ loại hình kinh doanh/dịch vụ
        _BUSINESS_WORDS = {
            "plaza", "tower", "center", "centre", "mall", "market",
            "coffee", "cafe", "hotel", "hostel", "restaurant", "clinic",
            "hospital", "pharmacy", "bank", "school", "university",
            "college", "church", "temple", "pagoda", "park", "garden",
            "station", "port", "airport", "embassy", "consulate",
            # tiếng Việt
            "tháp", "trung tâm", "siêu thị", "chợ", "bệnh viện",
            "trường", "đại học", "nhà thờ", "chùa", "công viên",
            "sân bay", "bến xe", "ga", "cảng", "đại sứ quán",
            "khách sạn", "nhà hàng", "quán", "tiệm", "cửa hàng",
        }
        words_set = set(no_accent.split())
        if not words_set.intersection(_BUSINESS_WORDS):
            # Kiểm tra thêm: tiền tố họ người Việt phổ biến dùng đặt tên đường
            _VN_SURNAME_PREFIXES = {
                "nguyen", "tran", "le", "pham", "huynh", "vo", "vu",
                "dang", "bui", "do", "ho", "ngo", "duong", "ly",
                "dinh", "truong", "phan", "luong", "chau", "luu",
                "mai", "to", "cao", "lam", "thai", "trinh", "nhan",
            }
            # Họ đứng đầu: rất có thể là tên đường dạng "Nguyễn Văn X"
            if words[0] in _VN_SURNAME_PREFIXES and len(words) >= 2:
                return True
            # Tiền tố nước ngoài phổ biến đặt tên đường tại VN
            _FOREIGN_PREFIXES = {
                "pasteur", "yersin", "calmette", "alexandre",
                "lyautey", "gallieni", "luro",
            }
            if no_accent.split()[0] in _FOREIGN_PREFIXES:
                return True

    return False


# ── JSON parser với fallback ──────────────────────────────────

def _parse_poi_response(text: str) -> Tuple[List[dict], bool]:
    """
    Parse kết quả từ response của LLM.
    Chỉ trích xuất JSON của danh sách POIs.
    """
    pois = []
    outside_district = False

    # 1. Kiểm tra cờ outside_district bằng regex
    if re.search(r"outside\s*:\s*true", text, re.IGNORECASE) or re.search(r'"outside_district"\s*:\s*true', text, re.IGNORECASE):
        outside_district = True

    # 2. Trích xuất JSON từ text nếu LLM xuất JSON (tìm cặp ngoặc nhọn hoặc vuông ngoài cùng)
    start_bracket = min(
        [idx for idx in [text.find('{'), text.find('[')] if idx != -1],
        default=-1
    )
    end_bracket = max(
        [idx for idx in [text.rfind('}'), text.rfind(']')] if idx != -1],
        default=-1
    )
    if start_bracket != -1 and end_bracket != -1 and end_bracket > start_bracket:
        json_str = text[start_bracket:end_bracket + 1]
        try:
            data = json.loads(json_str)
            if isinstance(data, list):
                pois_raw = data
            elif isinstance(data, dict):
                pois_raw = data.get("pois", [])
            else:
                pois_raw = []
                
            if isinstance(pois_raw, list):
                for p in pois_raw:
                    if isinstance(p, dict):
                        name = str(p.get("name", "")).strip()
                        name = name.strip("[]\"' ")
                        if name:
                            poi_dict = {
                                "name": name,
                            }
                            if "x" in p:
                                poi_dict["x"] = p["x"]
                            if "y" in p:
                                poi_dict["y"] = p["y"]
                            pois.append(poi_dict)
                    elif isinstance(p, str) and p.strip():
                        name = p.strip().strip("[]\"' ")
                        pois.append({
                            "name": name,
                        })
                if pois:
                    return pois, outside_district
        except json.JSONDecodeError:
            pass

    # 3. Fallback cuối: không tìm thấy POI nào
    logger.debug("No POI JSON found in LLM response.")
    return pois, outside_district
