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
    "Bạn là công cụ OCR dùng để đọc và sao chép chính xác các nhãn chữ trên ảnh bản đồ. "
    "Quy tắc tối thượng: chỉ ghi ra những chữ mà bạn đọc được rõ ràng từ ảnh. "
    "Tuyệt đối không tưởng tượng, dịch, hay bịa ra bất kỳ tên nào không có trong ảnh."
)

# ── Prompt ───────────────────────────────────────────────────
_PROMPT_TEMPLATE = """\
Ảnh này là ảnh chụp bản đồ OpenStreetMap của khu vực {district_name}.

Biến thể của ảnh:
- Kích thước ảnh: 256x256 pixel (toàn bộ ảnh là vùng quét cần phân tích).
- Ảnh có thể có khung nét đứt màu xanh Neon ở viền — bỏ qua, không liên quan.

Nhiệm vụ của bạn (làm theo THỨ TỰ này):

BƯỚC 1 — Đọc chữ trên ảnh:
- Nhìn vào từng vị trí trên ảnh và đọc từng nhãn chữ hiển thị (tên cửa tiệm, cửa hàng, ngân hàng, trường học, quán cafe, khách sạn v.v.).
- Sao chép NGUYÊN VĂN tên chữ như in trên bản đồ, kể cả chữ in hoa, dấu vầy, tiếng Việt hoặc tiếng Anh.
- Nếu không đọc được rõ ràng một tên → BUỘC PHẢI bỏ qua, không đoán.

BƯỚC 2 — Lọc địa điểm hợp lệ:
- CHỈ giữ lại địa điểm có TÊN RIÊNG cụ thể (ví dụ thực tế: "Bưu điện Trung tâm Sài Gòn", "Highlands Coffee", "Vincom Center", "Trường Tiểu học Lê Lợi").
- LOẠI BỎ hoàn toàn các nhãn thể loại chung không có tên riêng như: "Khách sạn", "Ngân hàng", "Siêu thị", "Café", "Nhà thờ", "Đường", "Công viên", "Nhà hàng", "Cửa hàng".
- LOẠI BỎ bất kỳ tên nào bạn TỰ NGHĨ RA hoặc suy diễn — chỉ ghi được tên nào bạn đọc được từ chữ trên bản đồ.

BƯỚC 3 — Ước lượng vị trí:
- Với mỗi địa điểm hợp lệ, ước lượng toạ độ x (0=trái, 100=phải) và y (0=trên, 100=dưới) trong ảnh.

BƯỚC 4 — Xác định ngoài quận:
- Nếu toàn bộ ảnh nằm RÕ RÀNG ngoài {district_name} → outside: true.
- Nếu nằm trong hoặc không chắc chắn → outside: false.

CÁCH TRẢ LỚI (bắt buộc dùng đúng định dạng này, không giải thích thêm):
- Bưu điện Trung tâm Sài Gòn (x: 45, y: 30)
- Highlands Coffee (x: 75, y: 80)
outside: false

Nếu không có địa điểm nào hợp lệ, chỉ cần trả lời:
outside: false

NHỚ LẠI LUẬT QUAN TRỌNG NHẤT:
✓ Chỉ ghi tên địa điểm nếu bạn ĐỌC ĐƯỢC chữ đó trực tiếp từ ảnh bản đồ.
✕ Không được viết "Cafe A", "Store B", "Restaurant C" hay bất kỳ tên mẫu nào — đây là vi phạm nghiêm trọng.
✕ Không tự dịch tên sang ngôn ngữ khác.
✕ Không đoán hoặc suy diễn tên từ icon hoặc biểu tượng.
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

            # Lọc bỏ các tên hallucination điển hình (Cafe A, Store B...)
            filtered_pois = []
            for p in unique_pois:
                if _is_hallucinated_name(p["name"]):
                    logger.warning(
                        "  [AntiHalluc] Bỏ tên nghi hallucination: '%s'", p["name"]
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
    # regex hỗ trợ bullet points khác nhau (-, *, numbered) và ký hiệu % tùy chọn
    pattern = r"(?:-|\*|\d+\.)\s*([^(]+?)\s*\(\s*x\s*:\s*(\d+)\s*%?\s*,\s*y\s*:\s*(\d+)\s*%?\s*\)"
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

    # 4. Fallback cuối: không tìm thấy POI nào — trả về rỗng thay vì đoán sắu
    # (Xóa fallback regex trích xuất ngoặc kép vốn gây hallucination)
    logger.debug("No POI pattern matched in LLM response.")
    return pois, outside_district
