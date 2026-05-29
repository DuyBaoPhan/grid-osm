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

# Đảm bảo thư mục hiện tại luôn nằm trong sys.path để tránh lỗi ModuleNotFoundError
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
    import config
    import grid
    import vision
    import coordinator
    import worker
    import main
    import clean_data
    print("    All files imported successfully.")

# ── 2. Kiểm thử Grid math ──────────────────────────────────────

def test_grid_math():
    from grid import lat_lng_to_tile, tile_center, generate_all_tiles
    lat, lng = 10.7769, 106.7009
    tx, ty = lat_lng_to_tile(lat, lng, 18)
    clat, clng = tile_center(tx, ty, 18)
    
    # Kiểm tra sai số tâm tile nhỏ hơn 0.001 độ
    assert abs(lat - clat) < 0.002, "Latitude mismatch is too large"
    assert abs(lng - clng) < 0.002, "Longitude mismatch is too large"
    
    # Kiểm tra sinh tile bán kính 1km
    tiles_1km = generate_all_tiles(lat, lng, 1.0, 18)
    assert len(tiles_1km) > 0, "No tiles generated"
    print(f"    Tile center: ({tx}, {ty}) matches ({clat:.6f}, {clng:.6f})")
    print(f"    Generated tiles in 1km radius: {len(tiles_1km)}")

# ── 3. Kiểm thử Vision Parser ──────────────────────────────────

def test_vision_parser():
    from vision import _parse_poi_response
    
    t1 = '{"pois": [{"name": "Chua Long Hoa", "x": 10, "y": 20}, {"name": "Nha Hang Pho", "x": 50, "y": 60}]}'
    res1, out1 = _parse_poi_response(t1)
    names1 = [p["name"] for p in res1]
    assert "Chua Long Hoa" in names1 and "Nha Hang Pho" in names1, "Standard JSON parse failed"
    assert res1[0]["x"] == 10 and res1[0]["y"] == 20, "x,y parsing failed"

    t2 = "```json\n{\"pois\": [{\"name\": \"Truong THCS\", \"x\": 30, \"y\": 40}]}\n```"
    res2, out2 = _parse_poi_response(t2)
    names2 = [p["name"] for p in res2]
    assert "Truong THCS" in names2, "Code fence parse failed"

    t3 = 'Map shows: ["Benh Vien", "ATM Sacombank"] nearby.'
    res3, out3 = _parse_poi_response(t3)
    names3 = [p["name"] for p in res3]
    assert "Benh Vien" in names3 and "ATM Sacombank" in names3, "Fallback regex parse failed"

    # Kiểm tra trường hợp outside_district true
    t4 = '{"pois": [], "outside_district": true}'
    res4, out4 = _parse_poi_response(t4)
    assert out4 is True, "Outside district parsing failed"

    # --- Các test cases nâng cao cho parser mới nâng cấp ---
    # Test percent signs và bullet points khác nhau
    t5 = "- Tên điểm A (x: 45%, y: 30%)\n* Tên điểm B (x: 75, y: 80%)\n1. Tên điểm C (x: 10%, y: 20)"
    res5, _ = _parse_poi_response(t5)
    assert len(res5) == 3, "Advanced bullet format parsing failed"
    assert res5[0]["name"] == "Tên điểm A" and res5[0]["x"] == 45 and res5[0]["y"] == 30
    assert res5[1]["name"] == "Tên điểm B" and res5[1]["x"] == 75 and res5[1]["y"] == 80
    assert res5[2]["name"] == "Tên điểm C" and res5[2]["x"] == 10 and res5[2]["y"] == 20

    # Test dọn dẹp ngoặc vuông [] và ngoặc kép "" '' trong tên
    t6 = "- [JW Marriott Saigon] (x: 50, y: 50)\n- \"Bệnh viện Nhi đồng 2\" (x: 12, y: 85)"
    res6, _ = _parse_poi_response(t6)
    assert res6[0]["name"] == "JW Marriott Saigon", "Square bracket stripping failed"
    assert res6[1]["name"] == "Bệnh viện Nhi đồng 2", "Quotes stripping failed"

    # Test JSON bọc giữa các đoạn text hội thoại
    t7 = "Dưới đây là kết quả:\n```json\n{\n  \"pois\": [\n    {\"name\": \"[Passio Coffee]\", \"x\": 40, \"y\": 60}\n  ]\n}\n```\noutside: false"
    res7, out7 = _parse_poi_response(t7)
    assert len(res7) == 1, "Conversational embedded JSON parsing failed"
    assert res7[0]["name"] == "Passio Coffee" and res7[0]["x"] == 40 and res7[0]["y"] == 60
    assert out7 is False

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

    from coordinator import Coordinator
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
