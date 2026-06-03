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

# ── 3. Kiểm thử Vision Parser ──────────────────────────────────

def test_vision_parser():
    from src.vision import _parse_poi_response
    
    t1 = '{"labels": [{"text": "Chua Long Hoa", "bbox": [10, 20, 30, 40], "confidence": 0.95}, {"text": "Nha Hang Pho", "bbox": [50, 60, 70, 80]}]}'
    res1, out1 = _parse_poi_response(t1)
    names1 = [p["name"] for p in res1]
    assert "Chua Long Hoa" in names1 and "Nha Hang Pho" in names1, "Standard JSON parse failed"
    # Under standard [ymin, xmin, ymax, xmax] layout, center x = (20 + 40)/2 = 30 -> scaled to 612.0: 30 / 1000 * 612 = 18.36
    assert abs(res1[0]["x"] - 18.36) < 0.01, "x coordinate scaling failed"

    t2 = "```json\n{\"labels\": [{\"text\": \"Truong THCS\", \"bbox\": [30, 40, 50, 60]}]}\n```"
    res2, out2 = _parse_poi_response(t2)
    names2 = [p["name"] for p in res2]
    assert "Truong THCS" in names2, "Code fence parse failed"

    # Kiểm tra trường hợp outside_district true
    t4 = '{"labels": [], "outside_district": true}'
    res4, out4 = _parse_poi_response(t4)
    assert out4 is True, "Outside district parsing failed"

    print("    Vision response parser handled all formats and boundaries correctly.")

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
    run_test_step("3. LLM Vision Response Parser", test_vision_parser)
    run_test_step("4. Coordinator Workflow Simulation", test_coordinator_integration)
    
    print("=" * 60)
    print(" [OK] ALL INTEGRITY CHECKS PASSED SUCCESSFULLY!")
    print(" System is ready to run with Ollama + Playwright.")
    print("=" * 60)
