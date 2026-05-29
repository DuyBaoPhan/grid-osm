# =============================================================
# OSM POI Scraper — config.py
# Toàn bộ hằng số cấu hình.
# Chỉnh sửa file này để điều chỉnh hành vi scraper.
# =============================================================

# ── Ollama / Local LLM ───────────────────────────────────────
OLLAMA_API_BASE   = "http://localhost:11434/v1"
OLLAMA_MODEL      = "qwen2.5vl:3b"           # Tên model 3B siêu nhẹ của Qwen2.5-VL

# ── Worker ───────────────────────────────────────────────────
NUM_WORKERS            = 2              # Để 1 worker cho độ ổn định tối đa (tránh quá tải GPU/VRAM khi gọi Ollama song song)
DELAY_BETWEEN_REQ      = 1.5            # giây nghỉ giữa mỗi tile
MAX_RETRIES            = 2              # số lần retry khi tile lỗi
BROWSER_RESTART_EVERY  = 100            # restart browser sau N tile
HEADLESS               = False          # Set False để hiển thị giao diện trình duyệt của từng worker

# ── Địa lý ───────────────────────────────────────────────────
ZOOM_LEVEL       = 18                   # zoom chia luoi tile (anh huong so luong tile)
SCREENSHOT_ZOOM  = 18                   # zoom hien thi trong URL browser (cang cao cang thay ro ten dia diem)
CENTER_LAT       = 10.779921962399342   # Tọa độ Bưu điện Thành phố làm trung tâm Quận 1
CENTER_LNG       = 106.70003066390481
RADIUS_KM        = 3.0                  # Bán kính 3km để bao phủ toàn bộ Quận 1
TARGET_DISTRICT  = "Quận 1"             # Tên quận cần quét để LLM tự động nhận diện biên giới và bỏ qua vùng ngoài quận

# ── Screenshot ───────────────────────────────────────────────
SCREENSHOT_W = 1024
SCREENSHOT_H = 768
SCREENSHOT_DIR = "screenshots"          # thư mục lưu ảnh debug (tùy chọn)
SAVE_SCREENSHOTS = True                # Đổi thành True để lưu ảnh xuống ổ đĩa!

# ── Map load ─────────────────────────────────────────────────
PAGE_LOAD_TIMEOUT  = 20_000            # ms — timeout goto()
PAGE_SETTLE_MS     = 1_500            # ms — chờ sau networkidle

# ── Expansion ────────────────────────────────────────────────
EXPAND_EMPTY = False                   # True = expand cả tile trống

# ── File I/O ─────────────────────────────────────────────────
CHECKPOINT_FILE = "checkpoint.json"
RESULTS_FILE    = "results.json"
LOG_FILE        = "scraper.log"

# ── Logging ──────────────────────────────────────────────────
LOG_LEVEL = "INFO"                     # DEBUG / INFO / WARNING
