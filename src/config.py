# =============================================================
# OSM POI Scraper — config.py
# Toàn bộ hằng số cấu hình.
# Chỉnh sửa file này để điều chỉnh hành vi scraper.
# =============================================================

from pathlib import Path

# Root directory của project (thư mục cha của src/)
BASE_DIR = Path(__file__).parent.parent

# ── Ollama / Local LLM ───────────────────────────────────────
OLLAMA_API_BASE   = "http://localhost:11434/v1"
OLLAMA_MODEL      = "qwen2.5vl:3b"           

# ── Worker ───────────────────────────────────────────────────
NUM_WORKERS            = 1              # Để 1 worker cho độ ổn định tối đa (tránh quá tải GPU/VRAM khi gọi Ollama song song)
DELAY_BETWEEN_REQ      = 1.5            # giây nghỉ giữa mỗi tile
MAX_RETRIES            = 2              # số lần retry khi tile lỗi
BROWSER_RESTART_EVERY  = 100            # restart browser sau N tile
HEADLESS               = False          # Set False để hiển thị giao diện trình duyệt của từng worker

# ── Địa lý ───────────────────────────────────────────────────
ZOOM_LEVEL       = 19                   # zoom chia luoi tile (anh huong so luong tile)
SCREENSHOT_ZOOM  = 19                   # zoom hien thi trong URL browser (cang cao cang thay ro ten dia diem)
CENTER_LAT       = 10.779930   # Tọa độ Bưu điện Thành phố làm trung tâm Quận 1
CENTER_LNG       = 106.699994
RADIUS_KM        = 3.0                  # Bán kính 3km để bao phủ toàn bộ Quận 1
TARGET_DISTRICT  = "Quận 1"             # Tên quận cần quét để LLM tự động nhận diện biên giới và bỏ qua vùng ngoài quận

# ── Screenshot / Tile (đã chuyển sang OSM Tile API) ────────────────────────────────────
OSM_TILE_URL   = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"  # Direct tile API
TILE_UPSCALE   = 2               # Upscale 256×256 → 512×512 trước khi gửi LLM
GRID_SIZE      = 5               # Lưới phân tích: 5×5 = 25 ô, mỗi ô ~3m×3m ở zoom 19
SCREENSHOT_W   = 1024            # Chỉ dùng cho DOM extraction (không còn crop tile)
SCREENSHOT_H   = 713
SCREENSHOT_DIR = str(BASE_DIR / "screenshots")  # thư mục lưu ảnh debug (tùy chọn)
SAVE_SCREENSHOTS = True          # Đổi thành True để lưu ảnh xuống ổ đĩa!

# ── Map load ─────────────────────────────────────────────────
PAGE_LOAD_TIMEOUT  = 20_000            # ms — timeout goto()
PAGE_SETTLE_MS     = 1_500            # ms — chờ sau networkidle

# ── Expansion ────────────────────────────────────────────────
EXPAND_EMPTY = False                   # True = expand cả tile trống

# ── File I/O ─────────────────────────────────────────────────
CHECKPOINT_FILE = str(BASE_DIR / "checkpoint.json")
RESULTS_FILE    = str(BASE_DIR / "results.json")
LOG_FILE        = str(BASE_DIR / "scraper.log")
STATUS_FILE     = str(BASE_DIR / "map_status.json")   # Dùng để HTML auto-reload khi có cập nhật

# ── Logging ──────────────────────────────────────────────────
LOG_LEVEL = "INFO"                     # DEBUG / INFO / WARNING

# ── OCR / VietOCR ─────────────────────────────────────────────
OCR_ENGINE = "vietocr"                 # vietocr | tesseract
VIETOCR_MODEL = "vgg_transformer"      # vgg_transformer | vgg_seq2seq
VIETOCR_DEVICE = "cpu"                 # cpu | cuda
OCR_TEXT_PAD_PX = 4                     # padding crop chữ trước khi nhận diện
OCR_ICON_MAX_Y_GAP = 48                 # icon phải nằm tối đa N px phía trên text
OCR_ICON_X_MARGIN = 18                  # icon được lệch ngang ngoài bbox text tối đa N px
