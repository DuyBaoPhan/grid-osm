# =============================================================
# demo_simulation.py — Giả lập tiến trình quét toàn bộ Quận 1
#
# Cách dùng:
#   python demo_simulation.py
#
# Ý nghĩa:
#   Giúp demo hiệu ứng "fog-of-war" quét tới đâu hiện tới đó
#   trên bản đồ Live HTML trong vòng 20-30 giây, thay vì phải
#   đợi chạy LLM thật mất nhiều giờ.
# =============================================================

import asyncio
import json
import logging
import os
import random
import time
import sys

# Điều hướng import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from coordinator import Coordinator
from grid import generate_all_tiles, tile_center

# Fix Windows terminal encoding (cp1252 không hỗ trợ tiếng Việt có dấu)
import io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("DemoSimulation")

# Ghi đè file lưu dữ liệu để tránh làm hỏng file kết quả thật của người dùng
config.CHECKPOINT_FILE = "checkpoint_demo.json"
config.RESULTS_FILE = "results_demo.json"
config.TARGET_DISTRICT = "Quận 1"

# Danh sách một số tên địa điểm ảo để giả lập kết quả POI phong phú
MOCK_POI_NAMES = [
    "Highlands Coffee", "The Coffee House", "Phúc Long Tea & Coffee",
    "Circle K", "FamilyMart", "GS25", "WinMart+", "Katinat Saigon Kafe",
    "Bánh mì Huỳnh Hoa", "Cửa hàng tiện lợi", "ATM Vietcombank", "ATM Techcombank",
    "Nhà sách Nguyễn Huệ", "Cà phê vỉa hè", "Phở Hòa Pasteur", "Bún chả Hoa Đông"
]

async def run_simulation():
    # Dọn dẹp checkpoint demo cũ để chạy mới tinh
    for f in [config.CHECKPOINT_FILE, config.RESULTS_FILE, config.CHECKPOINT_FILE + ".tmp", config.RESULTS_FILE + ".tmp"]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass

    logger.info("=" * 60)
    logger.info("  BẮT ĐẦU GIẢ LẬP TIẾN TRÌNH QUÉT QUẬN 1 (DEMO MODE)")
    logger.info("  Mục tiêu: Trực quan hóa lưới ô 'Quét tới đâu sáng tới đó'")
    logger.info("  Bản đồ hiển thị: map_viewer.html (với ranh giới xanh đậm Quận 1)")
    logger.info("=" * 60)

    # Khởi tạo coordinator
    coord = Coordinator()
    await coord.init()

    all_tiles = coord._all_tiles
    total_tiles = len(all_tiles)
    logger.info("Đã nạp %d ô tile thuộc Quận 1 từ đa giác ranh giới OSM.", total_tiles)

    if total_tiles == 0:
        logger.error("Không có tile nào trong đa giác Quận 1! Hãy kiểm tra file ranh giới.")
        return

    # Tự động mở map_viewer.html trên trình duyệt để người dùng xem trực tiếp
    map_url = "file:///" + os.path.abspath("map_viewer.html").replace(os.sep, "/")
    import webbrowser
    webbrowser.open(map_url)
    logger.info("Đã mở bản đồ theo dõi Live: %s", map_url)
    logger.info("Lời khuyên: Hãy đặt cửa sổ trình duyệt và cửa sổ Terminal cạnh nhau để xem hiệu ứng!")
    logger.info("Bắt đầu quét sau 3 giây...")
    await asyncio.sleep(3.0)

    start_time = time.time()
    idx = 0

    # Vòng lặp giống hệt worker thật trong main.py:
    #   get_next_tile() → xử lý → report_result() → coordinator tự expand neighbor
    # Không pre-sort, không biết trước thứ tự — hoàn toàn giống real crawl!
    while True:
        tile = await coord.get_next_tile()

        if tile is None:
            # Queue tạm trống — chờ một chút (giống worker.py)
            await asyncio.sleep(0.05)
            tile = await coord.get_next_tile()
            if tile is None:
                # Thực sự hết việc
                break

        idx += 1
        tx, ty = tile

        # 1. Lấy tọa độ tâm tile để sinh POI ảo
        clat, clng = tile_center(tx, ty, config.ZOOM_LEVEL)

        # 2. Giả lập thời gian worker chụp ảnh và phân tích
        delay = random.uniform(0.06, 0.15)
        await asyncio.sleep(delay)

        # 3. Tạo POI giả lập
        pois = []
        # Tỷ lệ 60% tìm thấy 1-3 địa điểm trong ô
        if random.random() < 0.60:
            num_pois = random.randint(1, 3)
            selected_names = random.sample(MOCK_POI_NAMES, num_pois)
            for name in selected_names:
                poi_lat = clat + random.uniform(-0.0002, 0.0002)
                poi_lng = clng + random.uniform(-0.0002, 0.0002)
                pois.append({
                    "name": f"{name} (Demo)",
                    "approx_lat": poi_lat,
                    "approx_lng": poi_lng,
                    "tile_x": tx,
                    "tile_y": ty,
                })

        # 4. Không giả lập outside_district — coordinator đã tự geo-verify
        #    chỉ tile trong polygon Quận 1 mới có trong queue
        outside_district = False

        # 5. Báo kết quả cho coordinator — tự động expand neighbor vào queue
        neighbors = [
            (tx + dx, ty + dy)
            for dx in [-1, 0, 1] for dy in [-1, 0, 1]
            if not (dx == 0 and dy == 0)
        ]
        await coord.report_result(tile, pois, neighbors, outside_district)

        # 6. Log tiến độ ra console
        total_done = len(coord._visited)
        total_tiles = len(coord._all_tiles)
        pct = (total_done / total_tiles) * 100 if total_tiles else 0
        speed = idx / (time.time() - start_time) * 60  # tiles/minute
        remaining = total_tiles - total_done
        eta_sec = remaining / (speed / 60) if speed > 0 else 0

        status_text = f"ĐÃ QUÉT (+{len(pois)} POIs)"
        logger.info(
            " Tiến trình: %d/%d (%d%%) | Ô (%d, %d) -> %s | Tốc độ: %.0f ô/phút | Còn lại: %.0fs",
            idx, total_tiles, int(pct), tile[0], tile[1], status_text, speed, eta_sec
        )

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("  HOÀN THÀNH GIẢ LẬP QUÉT QUẬN 1!")
    logger.info("  Tổng thời gian : %.1f giây", elapsed)
    logger.info("  Tổng số ô quét : %d", total_tiles)
    logger.info("  POIs giả lập   : %d địa điểm", len(coord._results))
    logger.info("  File kết quả   : %s", config.RESULTS_FILE)
    logger.info("=" * 60)
    logger.info("Hãy tải lại (F5) trang trình duyệt bản đồ để xem toàn bộ kết quả nhé!")

if __name__ == "__main__":
    try:
        asyncio.run(run_simulation())
    except KeyboardInterrupt:
        logger.warning("\n[!] Đã dừng giả lập bằng Ctrl+C.")
