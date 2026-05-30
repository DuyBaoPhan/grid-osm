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
    "Bạn là một công cụ OCR chuyên nghiệp để trích xuất tên địa điểm từ ảnh bản đồ. "
    "Nhiệm vụ: chỉ trích xuất các TÊN RIÊNG địa điểm thực tế (ví dụ: cửa hàng, cafe, bưu điện, tòa nhà, địa danh...). "
    "Quy tắc tối thượng: TUYỆT ĐỐI KHÔNG trích xuất tên đường phố, đường giao thông, đại lộ (như 'Hai Bà Trưng', 'Nguyễn Văn Bình', 'Lê Duẩn'...). "
    "Tuyệt đối không đoán, không tự bịa tên mẫu."
)

# ── Prompt ───────────────────────────────────────────────────
_PROMPT_TEMPLATE = """\
Đây là ảnh bản đồ của {district_name}.
Hãy đọc và trích xuất tất cả các tên riêng địa điểm tiếng Việt hiển thị trên ảnh bản đồ này.

QUY TẮC CỰC KỲ QUAN TRỌNG:
1. LOẠI BỎ hoàn toàn các tên đường phố, đường giao thông, đại lộ (ví dụ: "Hai Bà Trưng", "Nguyễn Văn Bình", "Lê Duẩn", "Đồng Khởi", "Đường...", "Street", "Rd", "Avenue"). Chỉ giữ lại điểm dịch vụ, cửa hiệu, địa danh du lịch, cơ quan.
2. Chỉ ghi những tên địa điểm (POI) bạn thực sự đọc được trên ảnh. Không đoán, không bịa tên mẫu (Ví dụ: KHÔNG được viết "Cafe A", "Store B").

Định dạng kết quả trả về bắt buộc (chỉ ghi kết quả này, không giải thích hay thêm bớt từ ngữ khác):
- Tên Địa Điểm (x: tọa độ x từ 0-100, y: tọa độ y từ 0-100)
outside: false
Ví dụ:
- Bưu điện Trung tâm Sài Gòn (x: 45, y: 60)
outside: false
"""


# ── Hàm chính ────────────────────────────────────────────────

async def extract_pois_from_screenshot(
    screenshot_bytes: bytes,
) -> Tuple[List[dict], bool]:
    """
    Gửi screenshot → Ollama → trả về tuple (list dict tên POI + vị trí, cờ outside_district).
    Tự động retry `MAX_RETRIES` lần nếu lỗi.
    """
    img_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
    prompt = _PROMPT_TEMPLATE.format(district_name=TARGET_DISTRICT)

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
                max_tokens=400,
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
                    unique_pois.append({
                        "name": name_clean,
                        "x": p.get("x", 50),
                        "y": p.get("y", 50)
                    })

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

def _is_hallucinated_name(name: str) -> bool:
    """Trả về True nếu tên trông như hallucination điển hình của LLM."""
    name = name.strip()
    if _HALLUC_PATTERN.match(name):
        return True
    # "[Từ bất kỳ] [Chữ cái đơn]" — ví dụ "Store B", "Bank A"
    if _SINGLE_LETTER_SUFFIX.search(name) and len(name.split()) <= 3:
        return True
    return False


# ── Bộ lọc tên đường giao thông (District 1 Streets Filter) ───

# Từ khóa liên quan đến giao thông để lọc bỏ
_STREET_KEYWORDS = re.compile(
    r"\b(đường|phố|đại lộ|boulevard|avenue|street|st\.|rd\.|hẻm|ngõ|kiệt|vòng xoay|cầu|ngã tư|ngã sáu|ngã bảy|ngã ba)\b",
    re.IGNORECASE
)

# Danh sách tên các con đường tại Quận 1 (được chuyển sang dạng không dấu để loại bỏ sai lệch dấu thanh/typo)
_D1_STREETS_NO_ACCENT = {
    "hai ba trung", "nguyen van binh", "le duan", "dong khoi", "ly tu trong",
    "le thanh ton", "nguyen hue", "pasteur", "nam ky khoi nghia", "ham nghi",
    "le loi", "nguyen thi minh khai", "ton duc thang", "cach mang thang tam",
    "tran hung dao", "nguyen trai", "bui vien", "pham ngu lao", "de tham",
    "nguyen du", "mac dinh chi", "phung khac khoan", "nguyen van chiem",
    "pham ngoc thach", "vo van tan", "nguyen thi nghia", "ton that dam",
    "huynh thuc khang", "nguyen cong tru", "pho duc chinh", "calmette",
    "nguyen thai hoc", "ky con", "yersin", "tran dinh xu", "nguyen cu trinh",
    "cong quynh", "mai thi luu", "dien bien phu", "nguyen binh khiem",
    "nguyen dinh chieu", "truong dinh", "ba huyen thanh quan",
    "suong nguyet anh", "ton that tung", "nguyen van trang", "nguyen thi dieu",
    "huyen tran cong chua", "nguyen trung truc", "thu khoa huan", "phan chu trinh",
    "phan boi chau", "nguyen an ninh", "le anh xuan",
    "ton that thiep", "ngo duc ke", "ho huan nghiep", "mac thi buoi",
    "nguyen thiep", "dong du", "thi sach", "thai van lung",
    "chu manh trinh", "nguyen sieu", "nguyen trung ngan",
    "vo thi sau", "thach thi thanh", "nguyen huu cau", "nguyen van thu",
    "tran cao van", "tran quang khai", "nguyen phi khanh",
    "dinh cong trang", "phan liem", "phan ke binh",
    "huynh khuong ninh", "pham viet chanh", "nguyen huu canh"
}

def _strip_vietnamese_accents(s: str) -> str:
    """Loại bỏ hoàn toàn dấu tiếng Việt và đưa về chữ thường."""
    s = s.lower()
    # a
    s = re.sub(r'[àáảãạăằắẳẵặâầấẩẫậ]', 'a', s)
    # e
    s = re.sub(r'[èéẻẽẹêềếểễệ]', 'e', s)
    # i
    s = re.sub(r'[ìíỉĩị]', 'i', s)
    # o
    s = re.sub(r'[òóỏõọôồốổỗộơờớởỡợ]', 'o', s)
    # u
    s = re.sub(r'[ùúủũụưừứửữự]', 'u', s)
    # y
    s = re.sub(r'[ỳýỷỹỵ]', 'y', s)
    # d
    s = re.sub(r'[đ]', 'd', s)
    return s

def _is_street_name(name: str) -> bool:
    """Trả về True nếu tên truyền vào khớp với tên đường giao thông."""
    name_clean = name.strip()
    name_lower = name_clean.lower()
    
    # 1. Nếu chứa từ khóa giao thông tiêu biểu -> Chắc chắn là tên đường
    if _STREET_KEYWORDS.search(name_lower):
        return True
        
    # 2. Loại bỏ tiền tố "đường", "phố", "đại lộ", "hẻm" nếu có để lấy tên lõi
    core_name = re.sub(r"^(đường|phố|đại lộ|hẻm|ngõ|kiệt)\s+", "", name_lower).strip()
    
    # 3. Chuyển sang không dấu để đối khớp danh sách đường Quận 1 một cách an toàn
    no_accent_name = _strip_vietnamese_accents(core_name)
    if no_accent_name in _D1_STREETS_NO_ACCENT:
        return True
        
    return False


# ── JSON parser với fallback ──────────────────────────────────

def _parse_poi_response(text: str) -> Tuple[List[dict], bool]:
    """
    Parse kết quả từ response của LLM.
    Hỗ trợ cả định dạng Text List mới và JSON fallback cũ.
    """
    pois = []
    outside_district = False

    # 1. Kiểm tra cờ outside_district bằng regex
    if re.search(r"outside\s*:\s*true", text, re.IGNORECASE) or re.search(r'"outside_district"\s*:\s*true', text, re.IGNORECASE):
        outside_district = True

    # 2. Bước 1: Thử parse theo định dạng Text List mới: "- Tên (x: 45, y: 30)" hoặc "* Tên (x: 45%, y: 30%)"
    # regex hỗ trợ dấu bullet point tùy chọn (?) giúp khớp cả trường hợp LLM không viết dấu gạch đầu dòng
    pattern = r"(?:-|\*|\d+\.)?\s*([^(]+?)\s*\(\s*x\s*:\s*(\d+)\s*%?\s*,\s*y\s*:\s*(\d+)\s*%?\s*\)"
    matches = re.findall(pattern, text)
    if matches:
        for m in matches:
            name = m[0].strip()
            # Dọn sạch các ký tự đặc biệt như ngoặc vuông, ngoặc kép, gạch ngang thừa
            name = re.sub(r"^-\s*", "", name)
            name = name.strip("[]\"' ")
            if name:
                pois.append({
                    "name": name,
                    "x": int(m[1]),
                    "y": int(m[2])
                })
        return pois, outside_district

    # 3. Bước 2: Trích xuất JSON từ text nếu LLM xuất JSON (tìm cặp ngoặc nhọn hoặc vuông ngoài cùng)
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
                            pois.append({
                                "name": name,
                                "x": p.get("x", 50),
                                "y": p.get("y", 50)
                            })
                    elif isinstance(p, str) and p.strip():
                        name = p.strip().strip("[]\"' ")
                        pois.append({
                            "name": name,
                            "x": 50,
                            "y": 50
                        })
                if pois:
                    return pois, outside_district
        except json.JSONDecodeError:
            pass

    # 4. Fallback cuối: không tìm thấy POI nào
    logger.debug("No POI pattern matched in LLM response.")
    return pois, outside_district
