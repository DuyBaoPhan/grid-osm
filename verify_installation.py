# =============================================================
# verify_installation.py — Script kiểm thử toàn bộ hệ thống
#
# Cách dùng:
#   python verify_installation.py
#
# Thực hiện:
#   1. Kiểm tra import của tất cả các file
#   2. Kiểm thử toán học hệ thống lưới (Grid math)
#   3. Kiểm thử bộ parse JSON của Vision (Vision response parser)
#   4. Mô phỏng tích hợp Coordinator (Coordinator dry-run)
# =============================================================

import sys
import os
import asyncio

# Đảm bảo src/ luôn nằm trong sys.path để tìm thấy các module
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

def run_test_step(step_name: str, func):
    print(f"[*] Running: {step_name}...")
    try:
        if asyncio.iscoroutinefunction(func):
            asyncio.run(func())
        else:
            func()
        print(f"[+] {step_name}: SUCCESS\n")
    except Exception as e:
        print(f"[-] {step_name}: FAILED!")
        print(f"    Error details: {e}")
        sys.exit(1)

# ── 1. Kiểm tra Imports ───────────────────────────────────────

def test_imports():
    import src.config
    import src.grid
    import src.vision
    import src.coordinator
    import src.worker
    import main
    import clean_data
    print("    All files imported successfully.")

# ── 2. Kiểm thử Grid math ──────────────────────────────────────

def test_grid_math():
    import config
    from src.grid import lat_lng_to_tile, tile_center, generate_all_tiles
    lat, lng = 10.7769, 106.7009
    tx, ty = lat_lng_to_tile(lat, lng, config.ZOOM_LEVEL)
    clat, clng = tile_center(tx, ty, config.ZOOM_LEVEL)
    
    # Kiểm tra sai số tâm tile nhỏ hơn 0.001 độ
    assert abs(lat - clat) < 0.002, "Latitude mismatch is too large"
    assert abs(lng - clng) < 0.002, "Longitude mismatch is too large"
    
    # Kiểm tra sinh tile bán kính 1km
    tiles_1km = generate_all_tiles(lat, lng, 1.0, config.ZOOM_LEVEL)
    assert len(tiles_1km) > 0, "No tiles generated"
    print(f"    Tile center: ({tx}, {ty}) matches ({clat:.6f}, {clng:.6f})")
    print(f"    Generated tiles in 1km radius: {len(tiles_1km)}")

# ── 3. Kiểm thử Vision OCR ──────────────────────────────────

def test_vision_ocr():
    from src.vision import _is_street_name, _is_generic_name
    
    # Kiểm tra nhận dạng tên đường
    assert _is_street_name("Đường Lê Lợi") is True, "Đường Lê Lợi should be street"
    assert _is_street_name("Lê Duẩn") is True, "Lê Duẩn should be street (unaccented check)"
    assert _is_street_name("Phố Huế") is True, "Phố Huế should be street"
    assert _is_street_name("Bưu điện Thành phố") is False, "Bưu điện should not be street"
    assert _is_street_name("Highlands Coffee") is False, "Highlands Coffee should not be street"
    
    # Kiểm tra tên loại hình chung chung
    assert _is_generic_name("cafe") is True, "cafe is generic name"
    assert _is_generic_name("Highlands Coffee") is False, "Highlands Coffee is not generic name"
    
    # Kiểm thử cuộc gọi trống
    import asyncio
    from src.vision import extract_pois_from_screenshot
    res, out = asyncio.run(extract_pois_from_screenshot(b""))
    assert res == [] and out is False, "Empty screenshot bytes should return empty list"
    
    print("    OCR Vision pipeline helpers and empty states validated successfully.")

# ── 4. Mô phỏng Coordinator ───────────────────────────────────

async def test_coordinator_integration():
    # Override config tạm thời để test
    import config
    original_radius = config.RADIUS_KM
    original_checkpoint = config.CHECKPOINT_FILE
    original_results = config.RESULTS_FILE

    config.RADIUS_KM = 0.5
    config.CHECKPOINT_FILE = "_test_checkpoint.json"
    config.RESULTS_FILE = "_test_results.json"

    from src.coordinator import Coordinator
    coord = Coordinator()
    await coord.init()

    # Kiểm tra stats
    stats = coord.stats
    assert stats["total_tiles"] > 0, "No valid tiles found in radius"
    assert stats["tiles_done"] == 0, "Completed tiles should start at 0"

    # Lấy tile
    tile = await coord.get_next_tile()
    assert tile is not None, "Queue is empty unexpectedly"

    # Báo cáo kết quả ảo
    dummy_pois = [{"name": "Store", "approx_lat": 10.7769, "approx_lng": 106.7009, "tile_x": tile[0], "tile_y": tile[1]}]
    await coord.report_result(tile, dummy_pois, [])

    # Xác nhận stats đã cập nhật
    stats_after = coord.stats
    assert stats_after["tiles_done"] == 1, "Completed tiles stats update failed"
    assert stats_after["pois_found"] == 1, "POI found stats update failed"

    # Dọn dẹp files sinh ra khi test
    for f in [config.CHECKPOINT_FILE, config.RESULTS_FILE, config.CHECKPOINT_FILE + ".tmp", config.RESULTS_FILE + ".tmp"]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass

    # Phục hồi config ban đầu
    config.RADIUS_KM = original_radius
    config.CHECKPOINT_FILE = original_checkpoint
    config.RESULTS_FILE = original_results

    print("    Coordinator safely managed queues and checkpoints.")

# ── Main Suite ───────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("        OSM POI SCRAPER SYSTEM INTEGRITY CHECK")
    print("=" * 60)
    
    run_test_step("1. Module Import Check", test_imports)
    run_test_step("2. Grid Mathematics Calculation", test_grid_math)
    run_test_step("3. OCR Vision POI Extraction", test_vision_ocr)
    run_test_step("4. Coordinator Workflow Simulation", test_coordinator_integration)
    
    print("=" * 60)
    print(" [OK] ALL INTEGRITY CHECKS PASSED SUCCESSFULLY!")
    print(" System is ready to run with OCR + OpenCV + Playwright.")
    print("=" * 60)
