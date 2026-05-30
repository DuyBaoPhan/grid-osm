# 🗺️ OSM POI Scraper — Local AI Edition

**OSM POI Scraper** là hệ thống khai thác địa điểm (POI - Point of Interest) tự động trên phạm vi hành chính cấp quận/huyện bằng cách kết hợp sức mạnh của **Playwright (Browser Automation)** và mô hình đa phương thức cục bộ **Qwen2.5-VL (Local AI Vision)** chạy qua **Ollama**. 

Dự án này được thiết kế để hoạt động **hoàn toàn ngoại tuyến (100% Local)**, không tốn bất kỳ chi phí API nào, có khả năng tự động vượt qua các rào cản cào dữ liệu thông thường của OpenStreetMap bằng cách "đọc" trực quan từ bản đồ thực tế.

---

## 🎨 Giao diện Bản đồ Real-time Dashboard
Hệ thống tự động đồng bộ hóa tiến trình và tạo bản đồ tương tác `map_viewer.html` trực quan:
*   **Xanh lá (Done):** Các ô lưới đã quét xong và tìm thấy POI.
*   **Xanh neon (Captured):** Các ô lưới đang được robot chụp ảnh và chuyển AI Vision phân tích.
*   **Sọc chéo (Discarded):** Các ô nằm ngoài ranh giới hành chính thực tế của Quận (Geofenced).
*   **Tự động Reload (Smart Auto-Reload):** Bản đồ tự nhận diện thay đổi trên đĩa và tải lại trang tự động tức thời nhờ cơ chế đồng bộ hóa mốc thời gian mili-giây (Mili-second Precision Timestamping), tương thích hoàn hảo cả khi mở bằng HTTP Server lẫn double-click trực tiếp file cục bộ (`file://` protocol) nhờ cơ chế bypass CORS động.

---

## 🚀 Tính năng nổi bật

### 1. Thuật toán Lưới dịch chuyển (Shifted Grid Math)
*   **Khớp tâm tuyệt đối (Epicenter Alignment):** Thay vì sử dụng lưới gạch OpenStreetMap tiêu chuẩn (thường bị lệch tọa độ tâm quét về một góc ngẫu nhiên), hệ thống tính toán sai số dịch vị (offset) vĩ độ/kinh độ so với lưới chuẩn:
    $$\Delta Lat = Lat_{Center} - Lat_{Tile\_Center}$$
    $$\Delta Lng = Lng_{Center} - Lng_{Tile\_Center}$$
*   **Đồng nhất bức chụp:** Tất cả các điểm quét được dịch chuyển đồng bộ, đảm bảo tâm quét của quận luôn trùng khớp **hoàn hảo 100%** với tâm của ô gạch trung tâm. Bức ảnh chụp bản đồ luôn cân đối, sắc nét.

### 2. Định vị Ranh giới thông minh (Geofencing & Ray Casting)
*   **Tự động tải boundary:** Nhập tên Quận (ví dụ: `Quận 1`), hệ thống tự tải ranh giới đa giác (Polygon/MultiPolygon) từ Nominatim API và lưu cache cục bộ dạng ASCII-safe để đảm bảo tốc độ cao nhất.
*   **Kiểm thử 5 điểm (5-Point Validation):** Để phủ kín hoàn hảo các ô lưới sát rìa quận, hệ thống sử dụng thuật toán **Ray Casting (Point-in-Polygon)** kiểm tra đồng thời tâm và 4 đỉnh góc của từng ô gạch. Chỉ cần 1 trong 5 điểm thuộc địa giới quận, ô đó sẽ được đưa vào danh sách quét.
*   **Lọc POI ngoại quận:** Tọa độ POI do AI Vision phân tích được đối chiếu với ranh giới quận để lọc bỏ ngay lập tức các kết quả bị "ảo tưởng" (hallucination) hoặc lấn sang địa giới quận lân cận.

### 3. Công cụ cào AI Vision mạnh mẽ (Vision & Playwright)
*   **Chất lượng ảnh độ nét cao:** Chụp màn hình bản đồ ở mật độ điểm ảnh cao (High-DPI Viewport), áp dụng bộ lọc tăng cường độ tương phản và sắc nét (**SHARPEN & Contrast**) giúp các nhãn văn bản nhỏ trên bản đồ trở nên cực kỳ dễ đọc đối với AI.
*   **Đa tiến trình song song (Multi-Worker Execution):** Hỗ trợ tối đa **4 worker song song** giả lập cào dữ liệu tốc độ cao. Cơ chế thay thế file an toàn (`_robust_replace`) triệt tiêu hoàn toàn lỗi khóa file trên hệ điều hành Windows (`[WinError 5] Access is denied`).
*   **Chống rò rỉ bộ nhớ (Memory Leak Protection):** Tự động khởi động lại trình duyệt Chromium ẩn (Headless) sau mỗi `100` ô quét nhằm giải phóng triệt để dung lượng RAM.
*   **Lưu Checkpoint nguyên tử:** Ghi checkpoint định kỳ thông qua cơ chế ghi file tạm rồi đổi tên (atomic rename), loại bỏ hoàn toàn khả năng hư hỏng tệp dữ liệu khi bị tắt đột ngột (Ctrl+C).

---

## 🛠️ Kiến trúc Hệ thống

```
main.py (Entry point)
  └── Coordinator (Quản lý queue, checkpoint, results)
        └── Worker(s) [1-4 workers song song]
              ├── grid.py       ← Xử lý toán học lưới và đa giác ranh giới
              ├── vision.py     ← Kết nối Ollama API & xử lý prompt đa phương thức
              └── config.py     ← Cấu hình hệ thống (tâm, bán kính, thông số quét)
```

---

## 💻 Yêu cầu Hệ thống

| Thành phần | Khuyến nghị |
|------------|-------------|
| **Hệ điều hành** | Windows 10/11, macOS, Linux |
| **Python** | Version 3.10 trở lên |
| **Ollama** | Phiên bản mới nhất (chạy nền) |
| **RAM** | $\ge$ 16 GB |
| **GPU/VRAM** | $\ge$ 6 GB VRAM để tăng tốc AI Vision (Ollama) |

---

## ⚙️ Cài đặt

```bash
# 1. Tải dự án về máy
git clone https://github.com/DuyBaoPhan/grid-osm.git
cd grid-osm

# 2. Tạo virtual environment & kích hoạt
python -m venv venv
# Trên Windows:
venv\Scripts\activate
# Trên macOS/Linux:
source venv/bin/activate

# 3. Cài đặt các thư viện cần thiết
pip install -r requirements.txt
playwright install chromium

# 4. Tải và chạy mô hình AI Vision qua Ollama
ollama run qwen2.5-vl
```

---

## 📖 Hướng dẫn sử dụng

### Bước 1: Cấu hình mục tiêu quét (`config.py`)
Mở file `config.py` để tùy chỉnh thông tin địa lý:
```python
# Tên quận để tải polygon ranh giới vẽ lên bản đồ và geofence
TARGET_DISTRICT = "Quận 1"

# Tọa độ tâm quét khởi điểm
CENTER_LAT = 10.7769
CENTER_LNG = 106.7009

# Bán kính quét (chỉ dùng làm fallback khi không tìm thấy polygon quận)
RADIUS_KM = 1.0

# Mức độ zoom (OSM Tile Zoom)
ZOOM_LEVEL = 18

# Số luồng chạy song song (1 luồng cho cào thực tế, tối đa 4 luồng cho giả lập)
NUM_WORKERS = 4
```

### Bước 2: Bắt đầu quét dữ liệu
Chạy tệp điều phối chính để khởi động quá trình:
```bash
python main.py
```
*Hệ thống sẽ tự động vẽ lưới đa giác quận, phân chia hàng đợi, khởi động Playwright cào dữ liệu và cập nhật trực tiếp tiến trình trên màn hình console cũng như bản đồ.*

### Bước 3: Xem bản đồ tương tác
Double-click trực tiếp file `map_viewer.html` trong thư mục dự án hoặc mở thông qua server local tại `http://127.0.0.1:8765/map_viewer.html`. Bản đồ sẽ **tự động tải lại** mỗi khi có tiến triển mới trên màn hình quét mà không cần bạn bấm F5 thủ công!

### Bước 4: Hậu xử lý & Kết xuất dữ liệu
Khi hoàn tất quét hoặc muốn trích xuất dữ liệu thô:
```bash
python clean_data.py
```
Kết quả được xuất ra 2 tệp sạch sẽ đã được khử trùng lặp (deduplicate) trong bán kính lân cận:
*   `clean_results.json`: Định dạng JSON lưu đầy đủ thông tin chi tiết.
*   `clean_results.csv`: Định dạng CSV sẵn sàng nạp vào Excel hoặc GIS software (QGIS, ArcGIS).

---

## 📊 Bảng Thống kê dự kiến (Quận 1, Zoom 18)

| Vùng quét | Số lượng ô lưới (Tiles) | Thời gian ước tính | Trạng thái |
|-----------|------------------------|--------------------|------------|
| **Quận 1 (Ranh giới)** | ~1.511 tiles | ~1.5 - 2 giờ (GPU) | Đầy đủ |
| **Thử nghiệm (1 km)** | ~138 tiles | ~10 - 15 phút | Test nhanh |
| **Bán kính 25 km** | ~87.000 tiles | ~3 - 5 ngày | Quy mô lớn |

---

## 📜 Giấy phép
Dự án được phân phối dưới giấy phép **MIT License**. Bạn được tự do tùy biến và sử dụng cho mục đích cá nhân và thương mại.
