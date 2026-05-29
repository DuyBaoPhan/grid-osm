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

# ── Prompt ───────────────────────────────────────────────────
_PROMPT_TEMPLATE = """\
Bạn là chuyên gia số hóa bản đồ chuyên nghiệp. Hãy phân tích ảnh chụp bản đồ OpenStreetMap này.

Bức ảnh này có kích thước 512x512 pixel. Ở CHÍNH GIỮA bức ảnh có một KHUNG QUÉT HÌNH VUÔNG MÀU XANH NEON (đường viền nét đứt màu xanh sáng, kích thước 256x256 pixel từ x=128 đến x=384, y=128 đến y=384 của ảnh).

Nhiệm vụ của bạn:
1. Hãy tìm và xác định vị trí của khung quét hình vuông màu xanh Neon ở chính giữa ảnh.
2. Đọc và trích xuất TẤT CẢ các địa điểm có tên riêng cụ thể (POI) có nhãn chữ hoặc biểu tượng (icon) nằm BÊN TRONG khung quét màu xanh Neon này.
   - Chỉ lấy địa điểm có tên riêng rõ ràng (ví dụ: tên cửa hàng, trường học, khách sạn, quán cafe cụ thể hiển thị bằng chữ trên bản đồ).
   - Tuyệt đối KHÔNG lấy các nhãn thể loại chung chung không có tên riêng (như: "Khách sạn", "Ngân hàng", "siêu thị", "Chùa", "Nhà thờ", "Cửa hàng", "Nhà hàng", "Đường phố", "Công viên").
   - Tuyệt đối KHÔNG lấy bất kỳ địa điểm nào nằm hoàn toàn ngoài khung quét màu xanh Neon.
3. Ước lượng tọa độ x, y (phần trăm từ 0 đến 100) của từng địa điểm ĐỐI VỚI KHUNG QUÉT MÀU XANH NEON:
   - x: 0 = cạnh trái khung Neon (x=128 của ảnh), 100 = cạnh phải khung Neon (x=384 của ảnh).
   - y: 0 = cạnh trên khung Neon (y=128 của ảnh), 100 = cạnh dưới khung Neon (y=384 của ảnh).
4. Xác định xem vị trí hiển thị trong ảnh này có nằm HOÀN TOÀN BÊN NGOÀI {district_name} hay không.
   Trả về "outside: true" nếu toàn bộ ảnh nằm ngoài {district_name}. Ngược lại trả về "outside: false".

Hãy trả về kết quả theo định dạng văn bản đơn giản sau (thay thế bằng tên và tọa độ thực tế tìm được, tuyệt đối không dùng các từ "Tên địa điểm" hay giải thích nào khác):
- Cafe A (x: 45, y: 30)
- Store B (x: 75, y: 80)
outside: false

Lưu ý đặc biệt quan trọng để tránh lỗi:
- Bạn phải trích xuất và ghi lại CHÍNH XÁC tên tiếng Việt hoặc tiếng Anh thực tế được ghi bằng chữ trên bản đồ (Ví dụ: "Bưu điện Trung tâm Sài Gòn", "Highlands Bưu Điện").
- TUYỆT ĐỐI KHÔNG tự bịa ra tên, không tự dịch nghĩa tiếng Việt sang tiếng Anh, và không đặt tên theo chuỗi ký tự A, B, C, D (Ví dụ: KHÔNG được ghi "Cafe A", "Store B", "Coffee Shop C", "Restaurant D" nếu trên bản đồ không thực sự có chữ đó).
- Chỉ trích xuất những địa điểm có nhãn chữ rõ ràng mà bạn đọc được trực tiếp từ ảnh. Nếu không đọc được chữ cụ thể, tuyệt đối bỏ qua.
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
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_b64}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=300,
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

            logger.debug("Vision OK: %d POIs extracted, outside=%s", len(unique_pois), outside)
            return unique_pois, outside

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

    # 4. Fallback cấp cuối cùng: trích xuất tất cả chữ trong ngoặc kép
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", text, flags=re.MULTILINE).strip()
    fallback = re.findall(r'"([^"]{2,80})"', cleaned)
    skip = {"pois", "outside_district", "name", "type", "lat", "lng", "x", "y"}
    for s in fallback:
        s_clean = s.strip().strip("[]\"' ")
        if s_clean and s_clean.lower() not in skip:
            pois.append({
                "name": s_clean,
                "x": 50,
                "y": 50
            })
    if pois:
        logger.debug("Used regex fallback, found %d items", len(pois))
        
    return pois, outside_district
