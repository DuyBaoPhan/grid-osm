# =============================================================
# Google Maps POI Scraper — config.py
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
ZOOM_LEVEL       = 21                   # zoom chia luoi tile - Google Maps max zoom = 21
SCREENSHOT_ZOOM  = 21                   # zoom hien thi trong URL browser - max 21 cho Google Maps
CENTER_LAT       = 10.779855797443227   # Tọa độ Bưu điện Thành phố làm trung tâm Quận 1
CENTER_LNG       = 106.69984398140998
RADIUS_KM        = 3.0                  # Bán kính 3km để bao phủ toàn bộ Quận 1
TARGET_DISTRICT  = "Quận 1"             # Tên quận cần quét để LLM tự động nhận diện biên giới và bỏ qua vùng ngoài quận

# ── Screenshot / Tile (đã chuyển sang Google Maps) ────────────────────────────────────
# Note: Google Maps không cung cấp public tile API như OSM
# MAP_TILE_URL   = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"  # Commented out - not used for Google Maps
TILE_UPSCALE   = 4               # Upscale 256×256 → 1024×1024 để text rõ hơn cho OCR
GRID_SIZE      = 5               # Lưới phân tích: 5×5 = 25 ô, mỗi ô ~3m×3m ở zoom 21
SCREENSHOT_W   = 1920            # Fullscreen width (1920x1080)
SCREENSHOT_H   = 1080            # Fullscreen height
SCREENSHOT_OVERLAP_PX = 80       # Khoảng tràn viền xung quanh ô quét (để tránh mất chữ/icon sát mép)
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
VIETOCR_BEAM_WIDTH = 20                # beam search width để khám phá nhiều khả năng nhận diện hơn
OCR_TEXT_PAD_PX = 24                   # padding crop chữ trước khi nhận diện (tăng từ 4 để có context tốt hơn)

# ── Icon-Text Matching (Google Maps horizontal layout) ────────
# User confirmed: Icon CÓ THỂ TRÁI hoặc PHẢI của text, khoảng cách ~5px, nằm ngang nhau
OCR_HORIZONTAL_GAP_MAX = 10            # Khoảng cách tối đa giữa icon và text theo phương ngang (pixels)
OCR_ICON_Y_ALIGN_RATIO = 1.3           # Icon Y phải nằm trong tỷ lệ này so với chiều cao text (flexible vì text có thể cao hơn nếu nhiều dòng)

# Deprecated (cho OSM vertical layout - icon phía trên text):
# OCR_ICON_MAX_Y_GAP = 24              # icon phải nằm tối đa N px phía trên text
# OCR_ICON_X_MARGIN = 12               # icon được lệch ngang ngoài bbox text tối đa N px
