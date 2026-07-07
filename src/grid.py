from __future__ import annotations

import math
from typing import List, Optional, Tuple

import config
from boundary_manager import load_geojson


def _center_frac(center_lat: float, center_lng: float, zoom: int) -> tuple[float, float, float]:
    n = 2 ** zoom
    cx = (center_lng + 180.0) / 360.0 * n
    lat_rad = math.radians(center_lat)
    cy = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return n, cx, cy


def lat_lng_to_tile(
    lat: float,
    lng: float,
    zoom: int,
    center_lat: Optional[float] = None,
    center_lng: Optional[float] = None,
) -> Tuple[int, int]:
    """Chuyển tọa độ địa lý → chỉ số grid custom gần nhất."""
    center_lat = config.CENTER_LAT if center_lat is None else center_lat
    center_lng = config.CENTER_LNG if center_lng is None else center_lng
    n, cx0, cy0 = _center_frac(center_lat, center_lng, zoom)
    cx_frac = (lng + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    cy_frac = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    step_x = config.SCREENSHOT_W / 256.0
    step_y = config.SCREENSHOT_H / 256.0
    return int(round((cx_frac - cx0) / step_x)), int(round((cy_frac - cy0) / step_y))


def tile_center(
    tx: int,
    ty: int,
    zoom: int,
    center_lat: Optional[float] = None,
    center_lng: Optional[float] = None,
) -> Tuple[float, float]:
    """Trả về tọa độ tâm custom grid cell."""
    center_lat = config.CENTER_LAT if center_lat is None else center_lat
    center_lng = config.CENTER_LNG if center_lng is None else center_lng
    n, cx0, cy0 = _center_frac(center_lat, center_lng, zoom)
    step_x = config.SCREENSHOT_W / 256.0
    step_y = config.SCREENSHOT_H / 256.0
    x_c = cx0 + tx * step_x
    y_c = cy0 + ty * step_y
    lng = x_c / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y_c / n))))
    return lat, lng


def tile_viewport_bbox(
    tx: int,
    ty: int,
    zoom: int,
    center_lat: Optional[float] = None,
    center_lng: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    """Trả về bbox thực tế của custom grid cell."""
    center_lat = config.CENTER_LAT if center_lat is None else center_lat
    center_lng = config.CENTER_LNG if center_lng is None else center_lng
    n, cx0, cy0 = _center_frac(center_lat, center_lng, zoom)
    step_x = config.SCREENSHOT_W / 256.0
    step_y = config.SCREENSHOT_H / 256.0
    x_c = cx0 + tx * step_x
    y_c = cy0 + ty * step_y
    x_min, x_max = x_c - step_x / 2.0, x_c + step_x / 2.0
    y_min, y_max = y_c - step_y / 2.0, y_c + step_y / 2.0

    def y_to_lat(y_f: float) -> float:
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y_f / n))))

    return (
        y_to_lat(y_max),
        x_min / n * 360.0 - 180.0,
        y_to_lat(y_min),
        x_max / n * 360.0 - 180.0,
    )


def tile_bbox(tx: int, ty: int, zoom: int) -> Tuple[float, float, float, float]:
    return tile_viewport_bbox(tx, ty, zoom)


def km_to_tile_radius(km: float, lat: float, zoom: int) -> int:
    tile_km = 40075.016 * math.cos(math.radians(lat)) / (2 ** zoom)
    return max(1, int(math.ceil(km / tile_km)))


def _safe_cache_path(district_name: str) -> str:
    from boundary_manager import slugify
    return f"boundary_{slugify(district_name)}.json"


def load_boundary_from_path(path: str) -> Optional[dict]:
    return load_geojson(path)


def load_or_download_boundary(district_name: str, province_name: str = "Thành phố Hồ Chí Minh") -> Optional[dict]:
    """Nạp/tải polygon ranh giới quận từ cache cũ ở root project."""
    import json
    import os
    import urllib.parse
    import urllib.request
    import logging

    logger = logging.getLogger(__name__)
    cache_path = _safe_cache_path(district_name)
    if os.path.exists(cache_path):
        geometry = load_geojson(cache_path)
        if geometry:
            logger.info("Loaded cached district boundary from %s", cache_path)
            return geometry

    query = f"{district_name}, {province_name}, Vietnam"
    url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(query)}&format=json&polygon_geojson=1&limit=1"
    logger.info("Downloading district boundary for '%s' from Nominatim API...", district_name)
    req = urllib.request.Request(url, headers={"User-Agent": "OSM-POI-Scraper/2.0 (+https://github.com/DuyBaoPhan/grid-osm)"})
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            results = json.loads(response.read().decode("utf-8"))
            if results and "geojson" in results[0]:
                geometry = results[0]["geojson"]
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(geometry, f, ensure_ascii=False, indent=2)
                return geometry
    except Exception as exc:
        logger.warning("Could not download boundary from Nominatim: %s", exc)
    return None


def is_point_in_boundary(lat: float, lng: float, geometry: dict, buffer_meters: float = 0.0) -> bool:
    """Kiểm tra điểm nằm trong GeoJSON Polygon/MultiPolygon, có buffer gần đúng."""
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates", [])

    def pip(x: float, y: float, poly: list) -> bool:
        inside = False
        n = len(poly)
        if n < 3:
            return False
        p1x, p1y = poly[0][0], poly[0][1]
        for i in range(n + 1):
            p2x, p2y = poly[i % n][0], poly[i % n][1]
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                xinters = p1x
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    rings: list[list] = []
    if geom_type == "Polygon" and coords:
        rings = [coords[0]]
    elif geom_type == "MultiPolygon":
        rings = [poly[0] for poly in coords if poly]
    if any(pip(lng, lat, ring) for ring in rings):
        return True
    if buffer_meters <= 0.0:
        return False

    lat_buf = buffer_meters / 111000.0
    lng_buf = buffer_meters / max(1.0, 111000.0 * math.cos(math.radians(lat)))
    max_buf_sq = max(lat_buf, lng_buf) ** 2

    def dist_sq(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
        dx, dy = bx - ax, by - ay
        if dx == 0 and dy == 0:
            return (px - ax) ** 2 + (py - ay) ** 2
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
        cx, cy = ax + t * dx, ay + t * dy
        return (px - cx) ** 2 + (py - cy) ** 2

    for ring in rings:
        for i, p1 in enumerate(ring):
            p2 = ring[(i + 1) % len(ring)]
            if dist_sq(lng, lat, p1[0], p1[1], p2[0], p2[1]) <= max_buf_sq:
                return True
    return False


def _geometry_bbox(geometry: dict) -> Optional[tuple[float, float, float, float]]:
    from boundary_manager import geometry_bbox
    bbox = geometry_bbox(geometry)
    if not bbox:
        return None
    lat_min, lng_min, lat_max, lng_max = bbox
    return lat_min, lng_min, lat_max, lng_max


def generate_tiles_for_boundary(center_lat: float, center_lng: float, zoom: int, geometry: dict) -> List[Tuple[int, int]]:
    import logging
    logger = logging.getLogger(__name__)
    bbox = _geometry_bbox(geometry)
    if not bbox:
        return []
    lat_min, lng_min, lat_max, lng_max = bbox
    n, cx0, cy0 = _center_frac(center_lat, center_lng, zoom)
    x1 = (lng_min + 180.0) / 360.0 * n
    x2 = (lng_max + 180.0) / 360.0 * n

    def lat_to_y(lt: float) -> float:
        return (1.0 - math.asinh(math.tan(math.radians(lt))) / math.pi) / 2.0 * n

    y1, y2 = lat_to_y(lat_min), lat_to_y(lat_max)
    step_x = config.SCREENSHOT_W / 256.0
    step_y = config.SCREENSHOT_H / 256.0
    tx_min = int(math.floor((min(x1, x2) - cx0) / step_x))
    tx_max = int(math.ceil((max(x1, x2) - cx0) / step_x))
    ty_min = int(math.floor((min(y1, y2) - cy0) / step_y))
    ty_max = int(math.ceil((max(y1, y2) - cy0) / step_y))
    tiles: List[Tuple[int, int]] = []
    for tx in range(tx_min, tx_max + 1):
        for ty in range(ty_min, ty_max + 1):
            clat, clng = tile_center(tx, ty, zoom, center_lat, center_lng)
            a, b, c, d = tile_viewport_bbox(tx, ty, zoom, center_lat, center_lng)
            points = [(a, b), (a, d), (c, b), (c, d), (clat, clng)]
            if any(is_point_in_boundary(lt, ln, geometry, 0.0) for lt, ln in points):
                tiles.append((tx, ty))
    logger.info("Generated %d tiles inside polygon boundary", len(tiles))
    return tiles


def generate_all_tiles(
    center_lat: float,
    center_lng: float,
    radius_km: float,
    zoom: int,
    district_name: Optional[str] = None,
    boundary_geometry: Optional[dict] = None,
) -> List[Tuple[int, int]]:
    if boundary_geometry:
        return generate_tiles_for_boundary(center_lat, center_lng, zoom, boundary_geometry)
    if district_name:
        geometry = load_or_download_boundary(district_name, getattr(config, "TARGET_PROVINCE", "Vietnam"))
        if geometry:
            return generate_tiles_for_boundary(center_lat, center_lng, zoom, geometry)
    tile_r = km_to_tile_radius(radius_km, center_lat, zoom)
    step_x = config.SCREENSHOT_W / 256.0
    step_y = config.SCREENSHOT_H / 256.0
    tx_r = int(math.ceil(tile_r / step_x))
    ty_r = int(math.ceil(tile_r / step_y))
    tiles: list[tuple[int, int]] = []
    for ty in range(-ty_r, ty_r + 1):
        for tx in range(-tx_r, tx_r + 1):
            clat, clng = tile_center(tx, ty, zoom, center_lat, center_lng)
            if _haversine(center_lat, center_lng, clat, clng) <= radius_km:
                tiles.append((tx, ty))
    return tiles


def _haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlng / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def pixel_to_gps(center_lat: float, center_lon: float, zoom: int, width: float, height: float, pixel_x: float, pixel_y: float, tile_size: int = 256) -> Tuple[float, float]:
    scale = tile_size * (2 ** zoom)
    center_world_x = (center_lon + 180.0) / 360.0 * scale
    sin_lat = max(min(math.sin(math.radians(center_lat)), 0.9999), -0.9999)
    center_world_y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * scale
    world_x = center_world_x + pixel_x - width / 2
    world_y = center_world_y + pixel_y - height / 2
    lon = world_x / scale * 360.0 - 180.0
    n = math.pi - (2 * math.pi * world_y / scale)
    lat = math.degrees(math.atan(math.sinh(n)))
    return lat, lon
