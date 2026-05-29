"""
map_viewer.py -- Tao ban do HTML hien thi trang thai cac tile da quet
"""

import json
import math
import os
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
from grid import generate_all_tiles, tile_bbox


# ── Doc du lieu hien tai ─────────────────────────────────────

def load_state():
    visited = set()
    queued  = set()
    discarded = set()

    if os.path.exists(config.CHECKPOINT_FILE):
        try:
            with open(config.CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            visited = {tuple(t) for t in data.get("visited", [])}
            queued  = {tuple(t) for t in data.get("queue",   [])}
            discarded = {tuple(t) for t in data.get("discarded", [])}
        except Exception as e:
            print(f"[WARN] Cannot load checkpoint: {e}")

    poi_count = 0
    if os.path.exists(config.RESULTS_FILE):
        try:
            with open(config.RESULTS_FILE, "r", encoding="utf-8") as f:
                results = json.load(f)
            poi_count = len(results)
        except Exception:
            pass

    return visited, queued, poi_count, discarded


# ── Tao GeoJSON polygons cho cac tile ────────────────────────

def build_geojson(all_tiles, visited, queued, discarded=None):
    if discarded is None:
        discarded = set()
    features = []
    for tile in all_tiles:
        tx, ty = tile

        # Chỉ hiển thị các ô đã được xử lý (visited hoặc discarded)
        # Ẩn hoàn toàn các ô chưa quét để tạo hiệu ứng "quét tới đâu hiện tới đó" cực đẹp!
        if tile not in visited and tile not in discarded:
            continue

        lat_min, lng_min, lat_max, lng_max = tile_bbox(tx, ty, config.ZOOM_LEVEL)

        # Xac dinh trang thai
        if tile in discarded:
            status = "discarded"
            color  = "url(#stripes)"  # Dung SVG stripes pattern
            opacity = 0.85
        elif tile in visited:
            status = "done"
            color  = "#00ff66"   # Xanh lá Neon phản quang cực sáng
            opacity = 0.70       # Tăng độ đậm đặc để nổi hẳn lên nền bản đồ
        else:
            continue

        feature = {
            "type": "Feature",
            "properties": {
                "tx": tx, "ty": ty,
                "status": status,
                "color": color,
                "opacity": opacity,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [lng_min, lat_min],
                    [lng_max, lat_min],
                    [lng_max, lat_max],
                    [lng_min, lat_max],
                    [lng_min, lat_min],
                ]],
            },
        }
        features.append(feature)

    return {"type": "FeatureCollection", "features": features}


# ── Sinh HTML ────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OSM POI Scraper -- Tile Map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@turf/turf@6/turf.min.js"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family: 'Segoe UI', sans-serif; background:#0f172a; color:#e2e8f0; }}
  #map {{ height: 100vh; width: 100%; }}

  #panel {{
    position: absolute; top: 16px; right: 16px; z-index: 1000;
    background: rgba(15,23,42,0.92);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 18px 22px;
    min-width: 250px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
  }}
  #panel h2 {{ font-size: 13px; font-weight: 700; color:#94a3b8; letter-spacing:.08em; text-transform:uppercase; margin-bottom:14px; }}

  .badge-live {{
    display:inline-flex; align-items:center; gap:5px;
    background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.15);
    border-radius:99px; padding:2px 10px; font-size:11px; color:#94a3b8;
    margin-bottom:14px;
  }}
  .dot-live {{ width:7px;height:7px;border-radius:50%;background:#94a3b8; }}

  .stat {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }}
  .stat-label {{ font-size:13px; color:#94a3b8; }}
  .stat-value {{ font-size:15px; font-weight:700; }}
  .val-done   {{ color:#22c55e; }}
  .val-queued {{ color:#f59e0b; }}
  .val-pending{{ color:#64748b; }}
  .val-poi    {{ color:#818cf8; }}
  .val-discarded {{ color:#f43f5e; }}

  .progress-bar {{
    background: #1e293b; border-radius: 99px;
    height: 8px; margin: 14px 0 4px;
    overflow: hidden;
  }}
  .progress-fill {{
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #22c55e, #38bdf8);
  }}
  .progress-label {{ font-size:11px; color:#64748b; text-align:right; }}

  .legend {{ margin-top:14px; border-top:1px solid rgba(255,255,255,0.08); padding-top:12px; }}
  .leg-item {{ display:flex; align-items:center; gap:8px; margin-bottom:5px; font-size:12px; color:#94a3b8; }}
  .leg-dot {{ width:12px; height:12px; border-radius:3px; flex-shrink:0; }}
  .tile-tooltip {{
    background: rgba(15,23,42,0.95);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 6px; color: #e2e8f0;
    font-size: 12px; padding: 4px 8px; white-space: nowrap;
  }}

  .btn-refresh {{
    display: block; width: 100%; margin-top: 14px;
    background: rgba(56, 189, 248, 0.12); border: 1px solid rgba(56, 189, 248, 0.4);
    color: #38bdf8; padding: 8px 16px; border-radius: 8px;
    font-size: 13px; font-weight: 600; cursor: pointer;
    transition: all 0.2s ease;
  }}
  .btn-refresh:hover {{
    background: #38bdf8; color: #0f172a;
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.4);
  }}
</style>
</head>
<body>

<!-- SVG Pattern definitions removed -->

<div id="map"></div>
<div id="panel">
  <div class="badge-live"><div class="dot-live"></div> Không tự làm mới (F5 để cập nhật)</div>
  <h2>Progress Overview</h2>
  <div class="stat"><span class="stat-label">Tổng số ô</span>      <span class="stat-value">{total}</span></div>
  <div class="stat"><span class="stat-label">Đã quét</span>       <span class="stat-value val-done">{done}</span></div>
  <div class="stat"><span class="stat-label">Bỏ qua (Ngoài quận)</span> <span class="stat-value val-discarded">{discarded}</span></div>
  <div class="stat"><span class="stat-label">Đang chờ (Queue)</span> <span class="stat-value val-queued">{queued}</span></div>
  <div class="stat"><span class="stat-label">Chưa xử lý</span>      <span class="stat-value val-pending">{pending}</span></div>
  <div class="stat"><span class="stat-label">Đã tìm thấy (POI)</span> <span class="stat-value val-poi">{pois}</span></div>

  <div class="progress-bar"><div class="progress-fill" style="width:{pct:.2f}%"></div></div>
  <div class="progress-label">{pct:.2f}% hoàn thành</div>

  <button onclick="window.location.reload()" class="btn-refresh">🔄 Làm mới bản đồ</button>

  <div class="legend">
    <div class="leg-item"><div class="leg-dot" style="background:#22c55e"></div> Đã quét (Done)</div>
    <div class="leg-item"><div class="leg-dot" style="background:#f59e0b"></div> Đang chờ xử lý (Queue)</div>
    <div class="leg-item"><div class="leg-dot" style="background:#475569"></div> Chưa bắt đầu (Pending)</div>
  </div>
</div>
<script>
// Khoi tao ban do voi mac dinh neu chua co saved state
let defaultCenter = [{center_lat}, {center_lng}];
let defaultZoom = {map_zoom};

const savedLat = localStorage.getItem('map_lat');
const savedLng = localStorage.getItem('map_lng');
const savedZoom = localStorage.getItem('map_zoom');

const map = L.map('map', {{
  center: (savedLat && savedLng) ? [parseFloat(savedLat), parseFloat(savedLng)] : defaultCenter,
  zoom: savedZoom ? parseInt(savedZoom) : defaultZoom
}});

L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  attribution: '&copy; OpenStreetMap contributors', maxZoom: 19,
}}).addTo(map);

L.circleMarker([{center_lat},{center_lng}], {{
  radius: 8, color: '#818cf8', fillColor: '#818cf8',
  fillOpacity: 0.9, weight: 2,
}}).addTo(map).bindPopup('<b>Tâm quét ({target_district})</b><br>{center_lat:.6f}, {center_lng:.6f}');

const boundaryGeojson = {boundary_geojson};
if (boundaryGeojson) {{
  L.geoJSON(boundaryGeojson, {{
    style: {{
      color: '#0055ff', // Màu xanh dương đậm cá tính và rõ nét
      weight: 5,        // Tăng độ dày viền lên 5px cực kỳ rõ nét
      fillColor: '#0055ff',
      fillOpacity: 0.08, // Màu nền đa giác đậm nét hơn một chút
      interactive: true
    }}
  }}).addTo(map).bindPopup('<b>Ranh giới hành chính: {target_district}</b>');
}}

const geojson = {geojson};
let processedGeojson = {{ type: "FeatureCollection", features: [] }};

if (typeof boundaryGeojson !== 'undefined' && boundaryGeojson && typeof turf !== 'undefined') {{
  const boundaryPoly = boundaryGeojson.geometry;
  
  geojson.features.forEach(f => {{
    try {{
      const tilePoly = f.geometry;
      
      // 1. Tính phần giao nhau (nằm TRONG ranh giới quận) -> Giữ nguyên thuộc tính Done
      const insideIntersection = turf.intersect(turf.feature(tilePoly), turf.feature(boundaryPoly));
      if (insideIntersection) {{
        const insideFeature = JSON.parse(JSON.stringify(f));
        insideFeature.geometry = insideIntersection.geometry;
        processedGeojson.features.push(insideFeature);
      }}

    }} catch (err) {{
      processedGeojson.features.push(f);
    }}
  }});
}} else {{
  processedGeojson = geojson;
}}

L.geoJSON(processedGeojson, {{
  style: f => ({{
    fillColor: f.properties.color, 
    fillOpacity: f.properties.color === 'url(#stripes)' ? 0.90 : 0.80, 
    color: f.properties.color === 'url(#stripes)' ? '#ff3355' : '#00ff66', 
    weight: 3.0, 
    opacity: 1.0, 
  }}),
  onEachFeature: (f, layer) => {{
    const p = f.properties;
    layer.bindTooltip(
      `Tile (${{p.tx}}, ${{p.ty}}) &mdash; <b>${{p.status === 'discarded' ? 'Ngoài ranh giới' : p.status}}</b>`,
      {{ className: 'tile-tooltip', sticky: true }}
    );
  }},
}}).addTo(map);

// Tu dong luu trang thai map khi keo, zoom
map.on('moveend', () => {{
  const center = map.getCenter();
  localStorage.setItem('map_lat', center.lat);
  localStorage.setItem('map_lng', center.lng);
}});
map.on('zoomend', () => {{
  localStorage.setItem('map_zoom', map.getZoom());
}});
</script>
</body>
</html>
"""

_MAP_OUT = os.path.join(os.path.dirname(__file__), "map_viewer.html")


def build_and_save(
    all_tiles,
    visited: set,
    queued: set,
    poi_count: int,
    discarded: set = None,
    out_path: str = _MAP_OUT,
) -> str:
    """
    Sinh map_viewer.html tu trang thai hien tai va ghi ra disk.
    Tra ve duong dan den file HTML.
    """
    if discarded is None:
        discarded = set()
    total   = len(all_tiles)
    done    = len(visited)
    q_count = len(queued)
    disc    = len(discarded)
    pending = max(0, total - done - q_count)
    pct     = done / total * 100 if total else 0

    # Đọc ranh giới hành chính của quận từ cache nếu có để vẽ lên bản đồ
    boundary_geojson_str = "null"
    from grid import _safe_cache_path
    cache_path = _safe_cache_path(config.TARGET_DISTRICT)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                boundary_data = json.load(f)
                boundary_feature = {
                    "type": "Feature",
                    "properties": {"name": config.TARGET_DISTRICT},
                    "geometry": boundary_data
                }
                boundary_geojson_str = json.dumps(boundary_feature)
        except Exception:
            pass

    geojson_str = json.dumps(build_geojson(all_tiles, visited, queued, discarded))
    display_zoom = max(10, config.ZOOM_LEVEL - 6)

    html = HTML_TEMPLATE.format(
        total=total, done=done, queued=q_count, discarded=disc,
        pending=pending, pois=poi_count, pct=pct,
        center_lat=config.CENTER_LAT,
        center_lng=config.CENTER_LNG,
        map_zoom=display_zoom,
        target_district=config.TARGET_DISTRICT,
        geojson=geojson_str,
        boundary_geojson=boundary_geojson_str,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


def main():
    visited, queued, poi_count, discarded = load_state()

    print("Generating tile grid...")
    all_tiles = generate_all_tiles(
        config.CENTER_LAT, config.CENTER_LNG,
        config.RADIUS_KM, config.ZOOM_LEVEL,
        config.TARGET_DISTRICT
    )

    total   = len(all_tiles)
    done    = len(visited)
    disc    = len(discarded)
    q_count = len(queued)
    pending = max(0, total - done - q_count)
    pct     = done / total * 100 if total else 0

    print(f"  Total  : {total}")
    print(f"  Done   : {done}")
    print(f"  Discard: {disc}")
    print(f"  Queued : {q_count}")
    print(f"  Pending: {pending}")
    print(f"  Done   : {pct:.2f}%")

    out_path = build_and_save(all_tiles, visited, queued, poi_count, discarded)
    print(f"\nMap saved: {out_path}")
    print("Opening in browser...")
    webbrowser.open(f"file:///{out_path.replace(os.sep, '/')}")


if __name__ == "__main__":
    main()

