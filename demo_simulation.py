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
    
    # Duyệt qua từng tile để giả lập quét
    for idx, tile in enumerate(all_tiles, 1):
        # 1. Lấy tọa độ tâm tile để sinh POI ảo
        clat, clng = tile_center(tile[0], tile[1], config.ZOOM_LEVEL)
        
        # 2. Giả lập thời gian worker chụp ảnh và phân tích (50 - 150ms để chạy demo nhanh mượt)
        delay = random.uniform(0.06, 0.15)
        await asyncio.sleep(delay)
        
        # 3. Tạo POI giả lập
        pois = []
        # Tỷ lệ 60% tìm thấy 1-3 địa điểm trong ô
        if random.random() < 0.60:
            num_pois = random.randint(1, 3)
            selected_names = random.sample(MOCK_POI_NAMES, num_pois)
            for name in selected_names:
                # Lệch nhẹ tọa độ xung quanh tâm tile để nhìn tự nhiên
                poi_lat = clat + random.uniform(-0.0002, 0.0002)
                poi_lng = clng + random.uniform(-0.0002, 0.0002)
                pois.append({
                    "name": f"{name} (Demo)",
                    "approx_lat": poi_lat,
                    "approx_lng": poi_lng,
                    "tile_x": tile[0],
                    "tile_y": tile[1]
                })

        # 4. Giả lập một số ít ô ở rìa sát biên giới bị coi là nằm ngoài quận (tỷ lệ 5%)
        # để hiển thị hiệu ứng gạch sọc đỏ cực đẹp
        outside_district = False
        if random.random() < 0.05:
            outside_district = True
            pois = []

        # 5. Khai báo kết quả lên coordinator
        # Hệ thống sẽ tự động ghi results_demo.json, checkpoint_demo.json và vẽ lại map_viewer.html
        neighbors = [
            (tile[0] + dx, tile[1] + dy)
            for dx in [-1, 0, 1] for dy in [-1, 0, 1]
            if not (dx == 0 and dy == 0)
        ]
        await coord.report_result(tile, pois, neighbors, outside_district)

        # 6. Log tiến độ ra console
        pct = (idx / total_tiles) * 100
        speed = idx / (time.time() - start_time) * 60  # tiles/minute
        eta_sec = (total_tiles - idx) / (speed / 60) if speed > 0 else 0
        
        status_text = "BỎ QUA (Biên)" if outside_district else f"ĐÃ QUÉT (+{len(pois)} POIs)"
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
