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

# ── Custom Grid coordinates for Perfect Edge-to-Edge Tiling ──

_n = 2 ** config.ZOOM_LEVEL
_cx_frac = (config.CENTER_LNG + 180.0) / 360.0 * _n
_lat_rad = math.radians(config.CENTER_LAT)
_cy_frac = (1.0 - math.asinh(math.tan(_lat_rad)) / math.pi) / 2.0 * _n


def lat_lng_to_tile(lat: float, lng: float, zoom: int) -> Tuple[int, int]:
    """Chuyển tọa độ địa lý → chỉ số grid (tx, ty) custom gần nhất."""
    n = 2 ** zoom
    cx_frac = (lng + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    cy_frac = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    
    step_x = config.SCREENSHOT_W / 256.0
    step_y = config.SCREENSHOT_H / 256.0
    
    tx = int(round((cx_frac - _cx_frac) / step_x))
    ty = int(round((cy_frac - _cy_frac) / step_y))
    return tx, ty


def tile_center(tx: int, ty: int, zoom: int) -> Tuple[float, float]:
    """
    Trả về tọa độ tâm của custom grid cell (tx, ty) dịch chuyển.
    tx: bước nhảy ngang (mỗi bước = 1024 px = 4.0 tile units)
    ty: bước nhảy dọc (mỗi bước = SCREENSHOT_H px = SCREENSHOT_H/256 tile units)
    """
    step_x = config.SCREENSHOT_W / 256.0
    step_y = config.SCREENSHOT_H / 256.0
    
    x_c = _cx_frac + tx * step_x
    y_c = _cy_frac + ty * step_y
    
    def _y_to_lat(y_f: float) -> float:
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y_f / _n))))
        
    lng = x_c / _n * 360.0 - 180.0
    lat = _y_to_lat(y_c)
    return lat, lng


def tile_viewport_bbox(tx: int, ty: int, zoom: int) -> Tuple[float, float, float, float]:
    """
    Trả về bounding box thực tế của custom grid cell (tx, ty)
    tiếp giáp khít mép (edge-to-edge) 0% gap và 0% overlap.
    """
    step_x = config.SCREENSHOT_W / 256.0
    step_y = config.SCREENSHOT_H / 256.0
    
    x_c = _cx_frac + tx * step_x
    y_c = _cy_frac + ty * step_y
    
    x_min = x_c - (step_x / 2.0)
    x_max = x_c + (step_x / 2.0)
    y_min = y_c - (step_y / 2.0)
    y_max = y_c + (step_y / 2.0)
    
    def _y_to_lat(y_f: float) -> float:
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y_f / _n))))
        
    lng_min = x_min / _n * 360.0 - 180.0
    lng_max = x_max / _n * 360.0 - 180.0
    lat_max = _y_to_lat(y_min)
    lat_min = _y_to_lat(y_max)
    
    return lat_min, lng_min, lat_max, lng_max


def tile_bbox(tx: int, ty: int, zoom: int) -> Tuple[float, float, float, float]:
    """Trả về bounding box giống tile_viewport_bbox."""
    return tile_viewport_bbox(tx, ty, zoom)


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
                
                # Chuyển boundary lat/lng sang fractional tile coordinates
                x1 = (lng_min + 180.0) / 360.0 * _n
                x2 = (lng_max + 180.0) / 360.0 * _n
                
                def _lat_to_y(lt: float) -> float:
                    return (1.0 - math.asinh(math.tan(math.radians(lt))) / math.pi) / 2.0 * _n
                
                y1 = _lat_to_y(lat_min)
                y2 = _lat_to_y(lat_max)
                
                x_min_frac, x_max_frac = min(x1, x2), max(x1, x2)
                y_min_frac, y_max_frac = min(y1, y2), max(y1, y2)
                
                step_x = config.SCREENSHOT_W / 256.0
                step_y = config.SCREENSHOT_H / 256.0
                
                # Tính phạm vi chỉ số tx, ty quanh tâm
                tx_min = int(math.floor((x_min_frac - _cx_frac) / step_x))
                tx_max = int(math.ceil((x_max_frac - _cx_frac) / step_x))
                ty_min = int(math.floor((y_min_frac - _cy_frac) / step_y))
                ty_max = int(math.ceil((y_max_frac - _cy_frac) / step_y))
                
                tiles: List[Tuple[int, int]] = []
                for tx in range(tx_min, tx_max + 1):
                    for ty in range(ty_min, ty_max + 1):
                        clat, clng = tile_center(tx, ty, zoom)
                        t_lat_min, t_lng_min, t_lat_max, t_lng_max = tile_viewport_bbox(tx, ty, zoom)
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
    tile_r = km_to_tile_radius(radius_km, center_lat, zoom)

    step_x = config.SCREENSHOT_W / 256.0
    step_y = config.SCREENSHOT_H / 256.0
    
    tx_r = int(math.ceil(tile_r / step_x))
    ty_r = int(math.ceil(tile_r / step_y))

    tiles = []
    for ty in range(-ty_r, ty_r + 1):
        for tx in range(-tx_r, tx_r + 1):
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


def pixel_to_gps(
    center_lat: float,
    center_lon: float,
    zoom: int,
    width: float,
    height: float,
    pixel_x: float,
    pixel_y: float,
    tile_size: int = 256
) -> Tuple[float, float]:
    """
    Chuyển pixel trên ảnh Google Maps/OSM -> GPS sử dụng Web Mercator Projection.
    """
    scale = tile_size * (2 ** zoom)

    # ===== GPS tâm -> World Pixel =====
    center_world_x = (center_lon + 180.0) / 360.0 * scale

    sin_lat = math.sin(math.radians(center_lat))
    sin_lat = max(min(sin_lat, 0.9999), -0.9999)

    center_world_y = (
        0.5
        - math.log((1 + sin_lat) / (1 - sin_lat))
        / (4 * math.pi)
    ) * scale

    # ===== Pixel tương đối so với tâm =====
    dx = pixel_x - width / 2
    dy = pixel_y - height / 2

    # ===== World Pixel của điểm cần tìm =====
    world_x = center_world_x + dx
    world_y = center_world_y + dy

    # ===== World Pixel -> GPS =====
    lon = world_x / scale * 360.0 - 180.0

    n = math.pi - (2 * math.pi * world_y / scale)

    lat = math.degrees(
        math.atan(
            math.sinh(n)
        )
    )

    return lat, lon


# ── Quick self-test ──────────────────────────────────────────
if __name__ == "__main__":
    zoom = 21
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
