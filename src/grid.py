# =============================================================
# grid.py — Hệ thống lưới OSM tile
#
# Cung cấp:
#   lat_lng_to_tile(lat, lng, zoom)       → (tx, ty)
#   tile_center(tx, ty, zoom)             → (lat, lng)
#   tile_bbox(tx, ty, zoom)               → (lat_min, lng_min, lat_max, lng_max)
#   generate_all_tiles(lat, lng, r_km, z) → list[(tx, ty)]
# =============================================================

import math
from typing import List, Tuple, Optional
import config

# ── Standard OSM tile math (Unshifted) ───────────────────────

def _std_lat_lng_to_tile(lat: float, lng: float, zoom: int) -> Tuple[int, int]:
    n = 2 ** zoom
    tx = int((lng + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    ty = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return tx, ty


def _std_tile_center(tx: int, ty: int, zoom: int) -> Tuple[float, float]:
    n = 2 ** zoom
    lng = (tx + 0.5) / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * (ty + 0.5) / n)))
    lat = math.degrees(lat_rad)
    return lat, lng


def _std_tile_bbox(tx: int, ty: int, zoom: int) -> Tuple[float, float, float, float]:
    n = 2 ** zoom

    def _y_to_lat(y_frac: float) -> float:
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y_frac / n))))

    lng_min = tx / n * 360.0 - 180.0
    lng_max = (tx + 1) / n * 360.0 - 180.0
    lat_max = _y_to_lat(ty)
    lat_min = _y_to_lat(ty + 1)
    return lat_min, lng_min, lat_max, lng_max


# ── Shift Calculation to center exactly on the Epicenter ──────
# We calculate the offset so that the tile containing (CENTER_LAT, CENTER_LNG)
# is centered exactly on (CENTER_LAT, CENTER_LNG).

_tx_c, _ty_c = _std_lat_lng_to_tile(config.CENTER_LAT, config.CENTER_LNG, config.ZOOM_LEVEL)
_lat_c, _lng_c = _std_tile_center(_tx_c, _ty_c, config.ZOOM_LEVEL)
OFFSET_LAT = config.CENTER_LAT - _lat_c
OFFSET_LNG = config.CENTER_LNG - _lng_c


# ── Shifted API functions ────────────────────────────────────

def lat_lng_to_tile(lat: float, lng: float, zoom: int) -> Tuple[int, int]:
    """Chuyển tọa độ địa lý → chỉ số tile OSM (tx, ty) dịch chuyển."""
    return _std_lat_lng_to_tile(lat - OFFSET_LAT, lng - OFFSET_LNG, zoom)


def tile_center(tx: int, ty: int, zoom: int) -> Tuple[float, float]:
    """Trả về tọa độ tâm của tile dịch chuyển (lat, lng)."""
    lat, lng = _std_tile_center(tx, ty, zoom)
    return lat + OFFSET_LAT, lng + OFFSET_LNG


def tile_bbox(tx: int, ty: int, zoom: int) -> Tuple[float, float, float, float]:
    """Trả về bounding box của tile dịch chuyển (lat_min, lng_min, lat_max, lng_max)."""
    lat_min, lng_min, lat_max, lng_max = _std_tile_bbox(tx, ty, zoom)
    return (
        lat_min + OFFSET_LAT,
        lng_min + OFFSET_LNG,
        lat_max + OFFSET_LAT,
        lng_max + OFFSET_LNG
    )


def tile_viewport_bbox(tx: int, ty: int, zoom: int) -> Tuple[float, float, float, float]:
    """
    Trả về bounding box thực tế của viewport worker (SCREENSHOT_W x SCREENSHOT_H CSS pixels)
    xoay quanh tâm của tile (tx, ty).
    """
    lat, lng = tile_center(tx, ty, zoom)
    
    n = 2 ** zoom
    
    cx_frac = (lng + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    cy_frac = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    
    dx = config.SCREENSHOT_W / 512.0
    dy = config.SCREENSHOT_H / 512.0
    
    x_min = cx_frac - dx
    x_max = cx_frac + dx
    y_min = cy_frac - dy
    y_max = cy_frac + dy
    
    def _y_to_lat(y_frac: float) -> float:
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y_frac / n))))
        
    lng_min = x_min / n * 360.0 - 180.0
    lng_max = x_max / n * 360.0 - 180.0
    lat_max = _y_to_lat(y_min)
    lat_min = _y_to_lat(y_max)
    
    return lat_min, lng_min, lat_max, lng_max


# ── Radius-based tile set ────────────────────────────────────

def km_to_tile_radius(km: float, lat: float, zoom: int) -> int:
    """
    Ước tính số tile tương ứng với khoảng cách km ở vĩ độ lat.
    1 tile ≈ 40075 * cos(lat) / 2^zoom km
    """
    tile_km = 40075.016 * math.cos(math.radians(lat)) / (2 ** zoom)
    return max(1, int(math.ceil(km / tile_km)))


def _safe_cache_path(district_name: str) -> str:
    """Loại bỏ dấu tiếng Việt để tạo tên file cache ASCII an toàn tuyệt đối trên Windows/Linux."""
    import unicodedata
    import re
    nfkd_form = unicodedata.normalize('NFKD', district_name)
    ascii_name = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    safe_name = "".join(c if c.isalnum() else "_" for c in ascii_name.lower())
    safe_name = re.sub(r"_+", "_", safe_name).strip("_")
    return f"boundary_{safe_name}.json"


def load_or_download_boundary(district_name: str) -> Optional[dict]:
    """
    Nạp polygon ranh giới quận từ cache file cục bộ.
    Nếu chưa có, tự động tải về từ OSM Nominatim API và lưu cache.
    """
    import json
    import os
    import urllib.parse
    import urllib.request
    import logging

    logger = logging.getLogger(__name__)
    
    # Chuẩn hóa tên file cache ASCII an toàn
    cache_path = _safe_cache_path(district_name)

    # 1. Nạp từ cache nếu tồn tại
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "type" in data and "coordinates" in data:
                    logger.info("Loaded cached district boundary from %s", cache_path)
                    return data
        except Exception as e:
            logger.warning("Could not read cached boundary file %s: %s", cache_path, e)

    # 2. Tải từ OSM Nominatim
    query = f"{district_name}, Thành phố Hồ Chí Minh, Vietnam"
    url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(query)}&format=json&polygon_geojson=1&limit=1"
    
    logger.info("Downloading district boundary for '%s' from Nominatim API...", district_name)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "OSM-POI-Scraper/2.0 (+https://github.com/DuyBaoPhan/grid-osm)"}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status == 200:
                results = json.loads(response.read().decode("utf-8"))
                if results and "geojson" in results[0]:
                    geometry = results[0]["geojson"]
                    # Ghi cache file
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(geometry, f, ensure_ascii=False, indent=2)
                    logger.info("Saved downloaded district boundary to %s", cache_path)
                    return geometry
                else:
                    logger.warning("No geojson geometry found in Nominatim response for '%s'", district_name)
    except Exception as e:
        logger.warning("Could not download boundary from Nominatim: %s", e)
        
    return None


def is_point_in_boundary(lat: float, lng: float, geometry: dict) -> bool:
    """
    Kiểm tra xem một tọa độ (lat, lng) có nằm trong GeoJSON geometry (Polygon hoặc MultiPolygon) không.
    Lưu ý: Tọa độ GeoJSON ở dạng [longitude, latitude].
    """
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates", [])

    def pip(x: float, y: float, poly: list) -> bool:
        inside = False
        n = len(poly)
        if n < 3:
            return False
        p1x, p1y = poly[0][0], poly[0][1] # lng, lat
        for i in range(n + 1):
            p2x, p2y = poly[i % n][0], poly[i % n][1]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    if geom_type == "Polygon":
        if not coords:
            return False
        return pip(lng, lat, coords[0])
    
    elif geom_type == "MultiPolygon":
        for poly_coords in coords:
            if poly_coords and pip(lng, lat, poly_coords[0]):
                return True
        return False
    
    return False


def generate_all_tiles(
    center_lat: float,
    center_lng: float,
    radius_km: float,
    zoom: int,
    district_name: Optional[str] = None,
) -> List[Tuple[int, int]]:
    """
    Sinh danh sách tile thuộc vùng cần quét.
    - Nếu cung cấp district_name và có thể tải/nạp polygon ranh giới:
      -> Trả về tất cả tile có tâm nằm trong boundary polygon của quận.
    - Nếu không, fall back về phương pháp sinh tile hình tròn theo bán kính radius_km truyền thống.
    """
    import logging
    logger = logging.getLogger(__name__)

    if district_name:
        geometry = load_or_download_boundary(district_name)
        if geometry:
            # Trích xuất bounding box của polygon để duyệt
            coords = geometry.get("coordinates", [])
            
            # Khởi tạo min/max lat/lng
            lats = []
            lngs = []
            
            def extract_points(lst):
                for item in lst:
                    if isinstance(item, list) and len(item) == 2 and isinstance(item[0], (int, float)):
                        lngs.append(item[0])
                        lats.append(item[1])
                    elif isinstance(item, list):
                        extract_points(item)
            
            extract_points(coords)
            
            if lats and lngs:
                lat_min, lat_max = min(lats), max(lats)
                lng_min, lng_max = min(lngs), max(lngs)
                
                logger.info(
                    "District polygon boundary bbox: lat=[%.6f, %.6f], lng=[%.6f, %.6f]",
                    lat_min, lat_max, lng_min, lng_max
                )
                
                tx1, ty1 = lat_lng_to_tile(lat_min, lng_min, zoom)
                tx2, ty2 = lat_lng_to_tile(lat_max, lng_max, zoom)
                tx_min, tx_max = min(tx1, tx2), max(tx1, tx2)
                ty_min, ty_max = min(ty1, ty2), max(ty1, ty2)
                
                logger.info(
                    "Generating tiles within district boundary grid box: tx=[%d, %d], ty=[%d, %d]",
                    tx_min, tx_max, ty_min, ty_max
                )
                
                tiles: List[Tuple[int, int]] = []
                for tx in range(tx_min, tx_max + 1):
                    for ty in range(ty_min, ty_max + 1):
                        clat, clng = tile_center(tx, ty, zoom)
                        # Phủ kín hoàn hảo ranh giới: chỉ cần tâm hoặc bất kỳ góc nào của ô
                        # nằm trong đa giác quận thì chấp nhận ô đó thuộc quận để quét.
                        t_lat_min, t_lng_min, t_lat_max, t_lng_max = tile_bbox(tx, ty, zoom)
                        corners = [
                            (t_lat_min, t_lng_min),
                            (t_lat_min, t_lng_max),
                            (t_lat_max, t_lng_min),
                            (t_lat_max, t_lng_max),
                            (clat, clng)
                        ]
                        if any(is_point_in_boundary(lat, lng, geometry) for lat, lng in corners):
                            tiles.append((tx, ty))
                
                logger.info("Generated %d tiles inside polygon boundary of '%s'", len(tiles), district_name)
                return tiles

    # Fallback to circle radius
    logger.info("Falling back to traditional circular radius-based tile generation.")
    cx_t, cy_t = lat_lng_to_tile(center_lat, center_lng, zoom)
    tile_r = km_to_tile_radius(radius_km, center_lat, zoom)

    tiles = []
    for dy in range(-tile_r, tile_r + 1):
        for dx in range(-tile_r, tile_r + 1):
            tx, ty = cx_t + dx, cy_t + dy
            if tx < 0 or ty < 0:
                continue
            clat, clng = tile_center(tx, ty, zoom)
            if _haversine(center_lat, center_lng, clat, clng) <= radius_km:
                tiles.append((tx, ty))

    return tiles


def _haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Khoảng cách Haversine tính bằng km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Quick self-test ──────────────────────────────────────────
if __name__ == "__main__":
    zoom = 18
    lat, lng = 10.7769, 106.7009
    tx, ty = lat_lng_to_tile(lat, lng, zoom)
    clat, clng = tile_center(tx, ty, zoom)
    print(f"Input  : ({lat}, {lng})")
    print(f"Tile   : ({tx}, {ty})")
    print(f"Center : ({clat:.6f}, {clng:.6f})")

    tiles_1km = generate_all_tiles(lat, lng, 1.0, zoom)
    tiles_25km = generate_all_tiles(lat, lng, 25.0, zoom)
    print(f"Tiles in  1 km radius: {len(tiles_1km)}")
    print(f"Tiles in 25 km radius: {len(tiles_25km)}")
