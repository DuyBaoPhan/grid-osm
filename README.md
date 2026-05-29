# OSM POI Scraper — Local AI Edition

Thu thập tên + tọa độ tất cả POI toàn TP.HCM từ OpenStreetMap,  
sử dụng **Playwright** + **Qwen2.5-VL** qua **Ollama** — chạy hoàn toàn Local, không tốn API.

---

## Kiến trúc

```
main.py
  └── Coordinator        ← quản lý queue, checkpoint, kết quả
        └── Worker(s)    ← Playwright browser + vision inference
              ├── grid.py     ← tile math OSM
              ├── vision.py   ← Ollama multimodal API
              └── config.py   ← tất cả hằng số cấu hình
```

---

## Yêu cầu

| Thành phần | Phiên bản |
|------------|-----------|
| Python     | 3.10+     |
| Ollama     | latest    |
| RAM        | ≥ 16 GB   |
| GPU VRAM   | ≥ 6 GB (tùy chọn, tăng tốc) |

---

## Cài đặt

```bash
# 1. Clone repo
git clone https://github.com/DuyBaoPhan/grid-osm.git
cd grid-osm

# 2. Cài Python dependencies
pip install -r requirements.txt
playwright install chromium

# 3. Cài & khởi động Ollama model
ollama run qwen2.5-vl
```

---

## Cách dùng

### Test với vùng nhỏ (1 km)

Mở `config.py` và đặt:
```python
RADIUS_KM = 1
```

```bash
python main.py
```

### Quét toàn TP.HCM (25 km)

```python
RADIUS_KM = 25   # ~87.000 tile
```

```bash
python main.py
```

### Hậu xử lý

```bash
python clean_data.py
# Xuất: clean_results.json + clean_results.csv
```

---

## Cấu trúc thư mục

```
osm-scraper/
├── main.py          ← entry point
├── coordinator.py   ← queue + checkpoint + results
├── worker.py        ← Playwright browser worker
├── grid.py          ← OSM tile math
├── vision.py        ← Ollama vision inference
├── config.py        ← tất cả cấu hình
├── clean_data.py    ← deduplicate + export
├── requirements.txt
├── checkpoint.json  ← tự tạo khi chạy
├── results.json     ← tự tạo khi chạy
└── scraper.log      ← log file
```

---

## Tính năng

- ✅ **Checkpoint nguyên tử** — atomic write (tmp → rename), không corrupt khi crash
- ✅ **Resume bất kỳ lúc nào** — Ctrl+C rồi chạy lại `python main.py`
- ✅ **Restart browser** mỗi 100 tile để giải phóng RAM leak
- ✅ **Retry logic** — thử lại tối đa `MAX_RETRIES` lần khi lỗi
- ✅ **JSON fallback** — parse kết quả LLM dù model không trả JSON chuẩn
- ✅ **ETA & progress** — log % hoàn thành + tốc độ tile/phút
- ✅ **Deduplicate** — khử trùng theo vùng tile ±1

---

## Cấu hình quan trọng (`config.py`)

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `RADIUS_KM` | `25` | Bán kính quét (km) |
| `ZOOM_LEVEL` | `18` | Zoom OSM (~150m/tile) |
| `DELAY_BETWEEN_REQ` | `1.5` | Giây nghỉ giữa tile |
| `BROWSER_RESTART_EVERY` | `100` | Restart browser sau N tile |
| `SAVE_SCREENSHOTS` | `False` | Lưu ảnh debug |
| `EXPAND_EMPTY` | `False` | Expand cả tile không có POI |

---

## Thống kê dự kiến

| Vùng | Số tile | Thời gian ước tính |
|------|---------|---------------------|
| 1 km (test) | ~138 tile | ~10 phút |
| 5 km | ~2.200 tile | ~2–3 giờ |
| 25 km (full) | ~87.000 tile | ~3–7 ngày |

> Thời gian phụ thuộc vào tốc độ GPU/CPU của Ollama.

---

## License

MIT
