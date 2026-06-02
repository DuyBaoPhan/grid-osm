# =============================================================
# clean_data.py — Hậu xử lý dữ liệu POI
#
# Cách dùng:
#   python clean_data.py
#   python clean_data.py --input results.json --output clean_results
#
# Thực hiện:
#   1. Load results.json
#   2. Chuẩn hóa tên (strip, collapse whitespace)
#   3. Khử trùng lặp theo tên + vùng tile (±1 tile)
#   4. Xuất clean_results.json + clean_results.csv
# =============================================================

import argparse
import csv
import json
import logging
import os
import re
import sys

# Đảm bảo src/ luôn nằm trong sys.path để tìm thấy các module
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Chuẩn hóa tên ────────────────────────────────────────────

def normalize_name(name: str) -> str:
    """
    Chuẩn hóa tên POI để so sánh trùng lặp:
      - Chuyển lowercase
      - Collapse whitespace
      - Loại bỏ ký tự đặc biệt thừa
    """
    name = name.strip().lower()
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"[\"\'`]", "", name)
    return name


# ── Deduplicate ───────────────────────────────────────────────

def deduplicate(raw_data: list) -> list:
    """
    Khử trùng lặp thông minh theo khoảng cách địa lý:
      - Cùng tên (đã normalize) và khoảng cách <= 200 mét -> coi là trùng lặp.
      - Giữ lại bản ghi đầu tiên gặp được.
      - Điều này giúp tránh lọc nhầm các chi nhánh cửa hàng/ATM cùng thương hiệu nhưng cách xa nhau.
    """
    import math
    
    def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        R = 6371000.0  # bán kính Trái Đất theo mét
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lng2 - lng1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    unique: list = []
    # Lưu dict map: name_key -> list of (lat, lng)
    seen_coords: dict = {}

    for item in raw_data:
        name_key = normalize_name(item.get("name", ""))
        if not name_key:
            continue

        lat = item.get("approx_lat")
        lng = item.get("approx_lng")
        
        # Nếu không có tọa độ, coi như không trùng
        if lat is None or lng is None:
            unique.append(item)
            continue

        is_duplicate = False
        if name_key in seen_coords:
            for (elat, elng) in seen_coords[name_key]:
                if haversine_m(lat, lng, elat, elng) <= 200.0:
                    is_duplicate = True
                    break

        if not is_duplicate:
            if name_key not in seen_coords:
                seen_coords[name_key] = []
            seen_coords[name_key].append((lat, lng))
            unique.append(item)

    return unique


# ── Xuất kết quả ─────────────────────────────────────────────

def export_json(data: list, path: str) -> None:
    # Chỉ giữ các field cần thiết
    clean = [
        {
            "name":             item["name"].strip(),
            "lat":              item.get("approx_lat"),
            "lng":              item.get("approx_lng"),
            "tile_x":           item.get("tile_x"),
            "tile_y":           item.get("tile_y"),
            "distance_pixels":  item.get("distance_pixels"),
            "distance_meters":  item.get("distance_meters"),
            "bearing_degrees":  item.get("bearing_degrees"),
        }
        for item in data
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
    logger.info("JSON exported → %s (%d records)", path, len(clean))


def export_csv(data: list, path: str) -> None:
    fieldnames = ["name", "lat", "lng", "tile_x", "tile_y", "distance_pixels", "distance_meters", "bearing_degrees"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for item in data:
            writer.writerow({
                "name":             item["name"].strip(),
                "lat":              item.get("approx_lat", ""),
                "lng":              item.get("approx_lng", ""),
                "tile_x":           item.get("tile_x", ""),
                "tile_y":           item.get("tile_y", ""),
                "distance_pixels":  item.get("distance_pixels", ""),
                "distance_meters":  item.get("distance_meters", ""),
                "bearing_degrees":  item.get("bearing_degrees", ""),
            })
    logger.info("CSV exported → %s (%d records)", path, len(data))


# ── Main ─────────────────────────────────────────────────────

def _haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    import math
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def geocode_poi(name: str, district: str = "Quận 1") -> list:
    """
    Sử dụng OpenStreetMap Nominatim API để tra cứu danh sách tọa độ của địa điểm.
    Trả về danh sách các dict {"lat", "lng"} trùng khớp.
    """
    import urllib.parse
    import urllib.request
    import json
    
    query = f"{name}, {district}, Thành phố Hồ Chí Minh, Vietnam"
    url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(query)}&format=json&limit=10"
    
    req = urllib.request.Request(
        url, 
        headers={"User-Agent": "OSM-POI-Scraper-Clean-Data/2.0 (+https://github.com/DuyBaoPhan/grid-osm)"}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                if data:
                    return [
                        {"lat": float(item["lat"]), "lng": float(item["lon"])}
                        for item in data
                        if "lat" in item and "lon" in item
                    ]
    except Exception as e:
        logger.debug("Geocoding failed for %s: %s", name, e)
    return []


def clean_and_deduplicate(
    input_file: str = "results.json",
    output_stem: str = "clean_results",
) -> None:
    if not os.path.exists(input_file):
        logger.error("Input file not found: %s", input_file)
        sys.exit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    logger.info("Raw records loaded: %d", len(raw_data))

    unique = deduplicate(raw_data)
    removed = len(raw_data) - len(unique)

    logger.info("After deduplication: %d unique POIs (removed %d duplicates)", len(unique), removed)

    # ── Tích hợp Geocoding thông minh giải quyết trùng lặp tên ──
    logger.info("Starting Geocoding lookup via OpenStreetMap Nominatim for exact coordinates...")
    geocoded_count = 0
    
    try:
        import config
        district = config.TARGET_DISTRICT
    except Exception:
        district = "Quận 1"
        
    for i, item in enumerate(unique, 1):
        name = item.get("name", "").strip()
        approx_lat = item.get("approx_lat")
        approx_lng = item.get("approx_lng")
        
        logger.info("[%d/%d] Geocoding: %s...", i, len(unique), name)
        
        # 1. Lấy danh sách tối đa 10 ứng viên trùng tên trong quận
        candidates = geocode_poi(name, district)
        
        exact_coords = None
        if candidates and approx_lat is not None and approx_lng is not None:
            # 2. Tìm ứng viên gần với vị trí ước lượng thị giác nhất
            best_cand = None
            min_dist = float("inf")
            
            for cand in candidates:
                dist = _haversine(approx_lat, approx_lng, cand["lat"], cand["lng"])
                if dist < min_dist:
                    min_dist = dist
                    best_cand = cand
            
            # 3. Chỉ chấp nhận nếu ứng viên gần nhất nằm trong phạm vi 150m (kích thước ô lưới)
            if best_cand and min_dist <= 0.15: # 0.15 km = 150 mét
                exact_coords = (best_cand["lat"], best_cand["lng"])
                logger.info("  [Success] Matched exact branch (dist=%.1fm): %.6f, %.6f", min_dist*1000.0, exact_coords[0], exact_coords[1])
            else:
                logger.info("  [Keep Visual] Closest candidate was too far (%.1fm)", min_dist*1000.0 if best_cand else 0)
        
        if exact_coords:
            item["approx_lat"] = exact_coords[0]
            item["approx_lng"] = exact_coords[1]
            
            # Tính toán lại distance_meters và bearing_degrees từ tọa độ Geocoding Nominatim mới so với tâm tile
            try:
                import math
                from grid import tile_center
                import config
                
                tx = item.get("tile_x")
                ty = item.get("tile_y")
                if tx is not None and ty is not None:
                    # Lấy tọa độ tâm tile
                    tile_lat, tile_lng = tile_center(tx, ty, config.ZOOM_LEVEL)
                    
                    # Tính khoảng cách thực địa (mét) và góc quay bearing (độ)
                    meters_per_degree_lat = 111132.9
                    meters_per_degree_lng = 111412.8 * math.cos(math.radians(tile_lat))
                    
                    dlat = exact_coords[0] - tile_lat
                    dlng = exact_coords[1] - tile_lng
                    
                    dy = dlat * meters_per_degree_lat
                    dx = dlng * meters_per_degree_lng
                    
                    new_dist = math.sqrt(dx**2 + dy**2)
                    new_bearing = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
                    
                    item["distance_meters"] = round(new_dist, 1)
                    item["bearing_degrees"] = round(new_bearing, 1)
                    
                    # Tính toán lại distance_pixels tương ứng với distance_meters mới
                    scale = 2.0  # Mặc định scale = 2.0
                    # Số mét trên mỗi CSS pixel ở vĩ độ hiện tại
                    R_earth = 6378137.0
                    tile_width_meters = (2 * math.pi * R_earth * math.cos(math.radians(tile_lat))) / (2 ** config.SCREENSHOT_ZOOM)
                    meters_per_css_pixel = tile_width_meters / 256.0
                    
                    # Quy đổi khoảng cách mét sang pixel vật lý
                    new_dist_css = new_dist / meters_per_css_pixel
                    new_dist_phys = new_dist_css * scale
                    item["distance_pixels"] = round(new_dist_phys, 1)
            except Exception as e:
                logger.debug("Failed to recalculate distance/bearing after geocoding: %s", e)
                
            geocoded_count += 1
        else:
            logger.info("  [Keep Visual] Keeping estimated coordinates: %.6f, %.6f", approx_lat, approx_lng)
            
        # Respect Nominatim Usage Policy (max 1 request per second)
        import time
        time.sleep(1.0)
        
    logger.info("Geocoding finished! Resolved and updated %d/%d exact coordinates.", geocoded_count, len(unique))

    export_json(unique, f"{output_stem}.json")
    export_csv(unique, f"{output_stem}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean & deduplicate OSM POI data")
    parser.add_argument("--input",  default="results.json",   help="Raw results file")
    parser.add_argument("--output", default="clean_results",  help="Output file stem (no extension)")
    args = parser.parse_args()

    clean_and_deduplicate(args.input, args.output)
