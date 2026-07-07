# Google Maps POI Scraper — Local AI OCR Edition

Hệ thống quét POI từ Google Maps theo lưới địa lý, dùng Playwright để chụp bản đồ, YOLOv8 để phát hiện nhãn POI, VietOCR/PaddleOCR để đọc chữ, và các lớp hậu xử lý tiếng Việt để chuẩn hóa tên địa điểm.

Dự án hiện tập trung vào bài toán: **tự động thu thập tên POI trong ranh giới hành chính**, lưu crop kiểm tra, định vị gần đúng từ pixel về tọa độ, khử trùng lặp và hiển thị tiến độ trên bản đồ local.

> Lưu ý: README cũ có nhắc Qwen/Ollama/OSM vision. Pipeline hiện tại đã chuyển sang Google Maps + YOLOv8 + VietOCR/PaddleOCR.

---

## Tính năng chính

### 1. Quét Google Maps theo lưới địa lý

- Sinh tile theo ranh giới quận/huyện.
- Dùng tâm cấu hình và zoom Google Maps (`ZOOM_LEVEL=21`).
- Chụp màn hình qua Playwright với viewport cố định.
- Crop vùng tile lõi từ screenshot lớn để tránh UI browser và mép bản đồ.
- Có checkpoint để tiếp tục sau khi dừng.

### 2. Phát hiện POI bằng YOLOv8

- Model: [model/bestv3.pt](file:///d:/grid-osm/model/bestv3.pt)
- Detect icon/label POI trên ảnh tile.
- Merge bbox gần/chồng nhau để không tách icon và text thành nhiều POI.
- Lọc bbox ngoài core tile để giảm POI trùng giữa các tile.

### 3. OCR local bằng VietOCR + PaddleOCR fallback

Pipeline chính nằm trong package [src/vision](file:///d:/grid-osm/src/vision):

- [detection.py](file:///d:/grid-osm/src/vision/detection.py): detect POI và điều phối OCR.
- [recognizers.py](file:///d:/grid-osm/src/vision/recognizers.py): VietOCR/PaddleOCR recognition.
- [geometry.py](file:///d:/grid-osm/src/vision/geometry.py): bbox expansion và text area detection.
- [crop_processing.py](file:///d:/grid-osm/src/vision/crop_processing.py): normalize crop, split line.
- [models.py](file:///d:/grid-osm/src/vision/models.py): lazy-load YOLO/VietOCR/PaddleOCR.
- [rendering.py](file:///d:/grid-osm/src/vision/rendering.py): debug draw và crop save.
- VietOCR đọc text chính.
- Chạy nhiều biến thể crop:
  - masked icon
  - unmasked icon
  - normalized background
  - line split
  - scale `1x/2x/3x`
- PaddleOCR dùng làm detector/recognition fallback khi VietOCR ra text rác.
- Lưu crop POI vào [crops](file:///d:/grid-osm/crops) để audit thủ công.

### 4. Hậu xử lý OCR tiếng Việt

Logic chính:

- [vietnam_places.py](file:///d:/grid-osm/src/vietnam_places.py)
- [ocr_language_corrections.json](file:///d:/grid-osm/data/ocr_language_corrections.json)
- [vietnam_places.txt](file:///d:/grid-osm/data/vietnam_places.txt)
- [osm_words.json](file:///d:/grid-osm/data/osm_words.json)

Các nhóm xử lý nằm trong [src/vision/text_cleaning](file:///d:/grid-osm/src/vision/text_cleaning):

- [dictionary.py](file:///d:/grid-osm/src/vision/text_cleaning/dictionary.py): accent/gazetteer/dictionary helpers.
- [spelling.py](file:///d:/grid-osm/src/vision/text_cleaning/spelling.py): spelling, duplicate token, diacritics merge.
- [junk.py](file:///d:/grid-osm/src/vision/text_cleaning/junk.py): junk/category filtering.
- [final_cleanup.py](file:///d:/grid-osm/src/vision/text_cleaning/final_cleanup.py): final OCR cleanup.
- [rescue.py](file:///d:/grid-osm/src/vision/text_cleaning/rescue.py): suffix/continuation rescue.
- [quality.py](file:///d:/grid-osm/src/vision/text_cleaning/quality.py): OCR quality, token checks, edge segment cleanup.

Nhóm xử lý chính:

- Chuẩn hóa địa danh Việt Nam theo gazetteer.
- Sửa dấu/chính tả bằng dictionary có context.
- Không sửa bừa brand/acronym như `GEOX`, `MCM`, `DAFC`.
- Xóa duplicate OCR:
  - `Ăn Ăn Vặt` → `Ăn Vặt`
  - `TP TP Hồ Chí Minh` → `TP Hồ Chí Minh`
- Cứu prefix/suffix khi OCR bị cắt:
  - giữ brand đầu như `Haeduri Hai Bà Trưng`
  - nối phần cuối tên khi alternate OCR có overlap hợp lệ
- Phát hiện chuỗi tiếng Việt giả để kích hoạt fallback:
  - ví dụ `Têm viên nông nân`

### 5. Dashboard bản đồ local

Các file liên quan:

- [map_viewer.html](file:///d:/grid-osm/map_viewer.html)
- [map_data.json](file:///d:/grid-osm/map_data.json)
- [map_status.json](file:///d:/grid-osm/map_status.json)
- [src/map_viewer.py](file:///d:/grid-osm/src/map_viewer.py)

Chức năng:

- Hiển thị tile đã quét.
- Hiển thị POI thu được.
- Tự reload khi dữ liệu thay đổi.
- Theo dõi tiến độ khi chạy [main.py](file:///d:/grid-osm/main.py).

---

## Kiến trúc thư mục

```text
grid-osm/
├─ main.py                         # Entry point
├─ requirements.txt                # Python dependencies
├─ checkpoint.json                  # Trạng thái quét hiện tại
├─ results.json                     # POI raw output
├─ map_data.json                    # Data cho map viewer
├─ map_viewer.html                  # Dashboard local
├─ scraper.log                      # Log runtime
├─ crops/                           # Crop POI OCR audit
├─ screenshots/                     # Screenshot tile/debug
├─ model/
│  └─ bestv3.pt                     # YOLOv8 model
├─ data/
│  ├─ vietnam_places.txt            # Gazetteer địa danh
│  ├─ osm_words.json                # Dictionary từ OSM
│  ├─ osm_raw_cache*.json           # Cache OSM raw
│  ├─ ocr_language_corrections.json # Rule sửa OCR có context
│  └─ ocr_ground_truth.json         # Ground truth/audit OCR
├─ src/
│  ├─ config.py                     # Cấu hình chính
│  ├─ coordinator.py                # Queue, checkpoint, dedupe, output
│  ├─ worker.py                     # Bản gốc đã comment để tham chiếu
│  ├─ worker/                       # Package Playwright worker runtime
│  │  ├─ __init__.py                 # Re-export Worker
│  │  ├─ worker.py                   # Class Worker + __init__
│  │  ├─ browser.py                  # Browser/page lifecycle
│  │  ├─ runner.py                   # Main worker loop
│  │  ├─ processing.py               # Tile processing + geo resolve
│  │  ├─ dom.py                      # DOM POI coordinate extraction
│  │  ├─ capture.py                  # Screenshot capture/save
│  │  └─ _original_worker_commented.py # Archive comment bản gốc
│  ├─ grid.py                       # Tile grid + boundary polygon
│  ├─ vision.py                     # Bản gốc đã comment để tham chiếu
│  ├─ vision/                       # Package YOLO + OCR runtime
│  │  ├─ __init__.py                 # API tương thích `from src.vision import ...`
│  │  ├─ models.py                   # Lazy-load model YOLO/VietOCR/PaddleOCR
│  │  ├─ detection.py                # Extract POI từ screenshot
│  │  ├─ recognizers.py              # VietOCR/PaddleOCR recognition
│  │  ├─ geometry.py                 # Bbox/text-area geometry
│  │  ├─ crop_processing.py          # Normalize/split OCR crop
│  │  ├─ rendering.py                # Draw detection + save crop
│  │  ├─ ocr_quality.py              # Quality helper re-export
│  │  ├─ text_cleaning/              # OCR cleanup package
│  │  │  ├─ __init__.py
│  │  │  ├─ dictionary.py
│  │  │  ├─ spelling.py
│  │  │  ├─ junk.py
│  │  │  ├─ final_cleanup.py
│  │  │  ├─ rescue.py
│  │  │  ├─ quality.py
│  │  │  └─ _original_text_cleaning_commented.py
│  │  └─ _original_vision_commented.py # Archive comment bản gốc
│  ├─ vietnam_places.py             # Vietnamese normalization
│  ├─ canonical_matcher.py          # Match/normalize tên chuẩn
│  └─ map_viewer.py                 # Local map data/server helper
└─ tests/
   └─ test_ocr_cleanup.py           # Regression tests OCR cleanup
```

---

## Yêu cầu hệ thống

| Thành phần | Khuyến nghị |
|---|---|
| OS | Windows 10/11 |
| Python | 3.10+ |
| RAM | >= 16 GB |
| GPU | Tùy chọn, OCR hiện cấu hình CPU |
| Browser | Chromium qua Playwright |
| Model | YOLOv8 weights trong `model/bestv3.pt` |

---

## Cài đặt

```powershell
git clone https://github.com/DuyBaoPhan/grid-osm.git
cd grid-osm

python -m venv venv
.\venv\Scripts\activate

pip install -r requirements.txt
playwright install chromium
```

Nếu dùng GPU cho torch/VietOCR, cài đúng bản PyTorch CUDA theo máy trước hoặc sau bước requirements.

---

## Cấu hình

Sửa [src/config.py](file:///d:/grid-osm/src/config.py).

Các cấu hình quan trọng:

```python
NUM_WORKERS = 1
HEADLESS = False

ZOOM_LEVEL = 21
SCREENSHOT_ZOOM = 21
CENTER_LAT = 10.779855797443227
CENTER_LNG = 106.69984398140998
RADIUS_KM = 3.0
TARGET_DISTRICT = "Quận 1"

SCREENSHOT_W = 1920
SCREENSHOT_H = 1080
SCREENSHOT_OVERLAP_PX = 100

SAVE_SCREENSHOTS = True
SAVE_POI_CROPS = True

VIETOCR_MODEL = "vgg_transformer"
VIETOCR_DEVICE = "cpu"
PADDLE_TEXT_DET_ENABLED = True

YOLO_MODEL_PATH = str(BASE_DIR / "model" / "bestv3.pt")
```

Khuyến nghị hiện tại:

- `NUM_WORKERS = 1` để ổn định khi mở Google Maps thật.
- `HEADLESS = False` để dễ quan sát browser.
- Giữ `SAVE_POI_CROPS = True` khi đang audit OCR.

---

## Chạy scraper

```powershell
python main.py
```

Hoặc nếu đang dùng Windows launcher:

```powershell
py main.py
```

Runtime tạo/cập nhật:

- [checkpoint.json](file:///d:/grid-osm/checkpoint.json)
- [results.json](file:///d:/grid-osm/results.json)
- [map_data.json](file:///d:/grid-osm/map_data.json)
- [scraper.log](file:///d:/grid-osm/scraper.log)
- [crops](file:///d:/grid-osm/crops)

---

## Xem dashboard

Mở:

```text
http://127.0.0.1:8765/map_viewer.html
```

Hoặc mở trực tiếp [map_viewer.html](file:///d:/grid-osm/map_viewer.html).

---

## Test

Chạy regression OCR cleanup:

```powershell
python -m pytest tests/test_ocr_cleanup.py
```

Test này bảo vệ các lỗi đã gặp:

- lặp từ: `Ăn Ăn`, `Gù Gù`, `TP TP`
- sai dấu/chính tả có context: `Nhà lẫm` → `Nhà Làm`
- brand/acronym không bị sửa sai: `GEOX`, `MCM`
- không xóa brand đầu dòng: `Haeduri Hai Bà Trưng`
- phát hiện OCR tiếng Việt giả: `Têm viên nông nân`
- cứu suffix/continuation tên địa điểm

Nên chạy test này trước khi sửa [src/vision](file:///d:/grid-osm/src/vision), [src/vision/text_cleaning](file:///d:/grid-osm/src/vision/text_cleaning), hoặc [vietnam_places.py](file:///d:/grid-osm/src/vietnam_places.py).

---

## Quy tắc sửa OCR trong dự án

Bắt buộc:

1. Không hardcode riêng cho một ảnh, một POI, một quận.
2. Tìm root cause trước khi sửa.
3. Mỗi bug OCR phải có regression test.
4. Không làm lỗi cũ xuất hiện lại.
5. Rule phải tổng quát cho dữ liệu Việt Nam.
6. Brand/acronym phải được bảo vệ.
7. Dictionary correction phải có context.

Nơi thêm rule:

- Rule ngôn ngữ có context: [ocr_language_corrections.json](file:///d:/grid-osm/data/ocr_language_corrections.json)
- Gazetteer địa danh: [vietnam_places.txt](file:///d:/grid-osm/data/vietnam_places.txt)
- Logic detect/OCR: [src/vision](file:///d:/grid-osm/src/vision)
- Logic cleanup/selection OCR: [src/vision/text_cleaning](file:///d:/grid-osm/src/vision/text_cleaning)
- Normalize từ/phrase: [vietnam_places.py](file:///d:/grid-osm/src/vietnam_places.py)

---

## Hậu xử lý dữ liệu

Nếu cần làm sạch kết quả sau khi quét:

```powershell
python clean_data.py
```

Các output có thể gồm:

- `clean_results.json`
- `clean_results.csv`

Tùy script hiện tại và cấu hình output.

---

## Debug nhanh

### Xem log OCR

```powershell
Select-String -Path scraper.log -Pattern "OCR"
```

### Chạy lại test OCR

```powershell
python -m pytest tests/test_ocr_cleanup.py -q
```

### Audit crop

Mở thư mục:

```text
crops/
```

Tên file crop thường chứa text OCR cuối, ví dụ:

```text
tile_0_0_poi_8_Cổng_Đường_sách__TP_TP_Hồ_Chí_Minh.png
```

---

## License

MIT License.
