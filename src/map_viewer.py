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
from grid import generate_all_tiles, tile_bbox, tile_viewport_bbox


# ── Doc du lieu hien tai ─────────────────────────────────────

def load_state():
    visited = set()
    queued  = set()
    discarded = set()
    captured = set()
    results = []

    if os.path.exists(config.CHECKPOINT_FILE):
        try:
            with open(config.CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            visited = {tuple(t) for t in data.get("visited", [])}
            queued  = {tuple(t) for t in data.get("queue",   [])}
            discarded = {tuple(t) for t in data.get("discarded", [])}
            captured = {tuple(t) for t in data.get("captured", [])}
        except Exception as e:
            print(f"[WARN] Cannot load checkpoint: {e}")

    if os.path.exists(config.RESULTS_FILE):
        try:
            with open(config.RESULTS_FILE, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            pass

    return visited, queued, results, discarded, captured


# ── Tao GeoJSON polygons cho cac tile ────────────────────────

def build_geojson(all_tiles, visited, queued, discarded=None, captured=None):
    if discarded is None:
        discarded = set()
    if captured is None:
        captured = set()
    features = []
    for tile in all_tiles:
        tx, ty = tile

        # Chỉ hiển thị các ô đã được xử lý (visited hoặc discarded) hoặc đã chụp (captured)
        if tile not in visited and tile not in discarded and tile not in captured:
            continue

        lat_min, lng_min, lat_max, lng_max = tile_viewport_bbox(tx, ty, config.ZOOM_LEVEL)

        # Xac dinh trang thai
        if tile in discarded:
            status = "discarded"
            color  = "url(#stripes)"  # Dung SVG stripes pattern
            opacity = 0.85
        elif tile in visited:
            status = "done"
            color  = "#00ff66"   # Xanh lá Neon phản quang cực sáng
            opacity = 0.70       # Tăng độ đậm đặc để nổi hẳn lên nền bản đồ
        elif tile in captured:
            status = "captured"
            color  = "#00d2ff"   # Xanh Neon Blue phản quang cho ô đang cào
            opacity = 0.70
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
  .dot-live {{ width:7px;height:7px;border-radius:50%;background:#94a3b8;transition:background 0.3s; }}
  .dot-live.active {{ background:#22c55e; animation: pulse 1.5s infinite; }}
  @keyframes pulse {{
    0%, 100% {{ box-shadow: 0 0 0 0 rgba(34,197,94,0.5); }}
    50% {{ box-shadow: 0 0 0 5px rgba(34,197,94,0); }}
  }}

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
  <div class="badge-live"><div class="dot-live"></div> <span id="live-status">Đang kết nối...</span></div>
  <h2>Progress Overview</h2>
  <div class="stat"><span class="stat-label">Tổng số ô</span>      <span class="stat-value">{total}</span></div>
  <div class="stat"><span class="stat-label">Đã quét</span>       <span class="stat-value val-done">{done}</span></div>
  <div class="stat"><span class="stat-label">Đang chờ</span> <span class="stat-value val-queued">{queued}</span></div>
  <div class="stat"><span class="stat-label">Đã tìm thấy</span> <span class="stat-value val-poi">{pois}</span></div>

  <div class="progress-bar"><div class="progress-fill" style="width:{pct:.2f}%"></div></div>
  <div class="progress-label">{pct:.2f}% hoàn thành</div>

  <button onclick="window.location.reload()" class="btn-refresh">🔄 Làm mới bản đồ</button>

  <div class="legend">
    <div class="leg-item"><div class="leg-dot" style="background:#22c55e"></div> Đã quét (Done)</div>
    <div class="leg-item"><div class="leg-dot" style="background:#00d2ff"></div> Đang xử lý AI (Captured)</div>
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
let geoJsonLayer = null;

function renderGeoJson(geoJsonData) {{
  if (geoJsonLayer) {{
    map.removeLayer(geoJsonLayer);
  }}

  let processedGeojson = {{ type: "FeatureCollection", features: [] }};

  if (typeof boundaryGeojson !== 'undefined' && boundaryGeojson && typeof turf !== 'undefined') {{
    const boundaryPoly = boundaryGeojson.geometry;
    
    geoJsonData.features.forEach(f => {{
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
    processedGeojson = geoJsonData;
  }}

  geoJsonLayer = L.geoJSON(processedGeojson, {{
    style: f => ({{
      fillColor: f.properties.color, 
      fillOpacity: f.properties.color === 'url(#stripes)' ? 0.90 : 0.80, 
      color: f.properties.color === 'url(#stripes)' ? '#ff3355' : (f.properties.status === 'captured' ? '#00d2ff' : '#00ff66'), 
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
}}

// Vẽ lưới ban đầu
renderGeoJson(geojson);

// Vẽ các địa điểm (POIs) đã tìm thấy
const poisData = {pois_data};
const poiLayerGroup = L.layerGroup().addTo(map);

function renderPois(poisList) {{
  poiLayerGroup.clearLayers();
  poisList.forEach(poi => {{
    L.circleMarker([poi.lat, poi.lng], {{
      radius: 6,
      color: '#0f172a',
      weight: 1.5,
      opacity: 1.0,
      fillColor: '#f43f5e',
      fillOpacity: 0.95
    }}).addTo(poiLayerGroup).bindPopup(`
      <div style="font-family: 'Segoe UI', sans-serif; color: #0f172a; padding: 2px 0; min-width: 150px;">
        <span style="font-size: 10px; font-weight: 700; color: #f43f5e; text-transform: uppercase; letter-spacing: 0.05em;">${{poi.type || 'Địa điểm'}}</span>
        <h4 style="margin: 4px 0 2px 0; font-size: 13px; font-weight: 700; line-height: 1.3;">${{poi.name}}</h4>
        <span style="font-size: 10px; color: #64748b;">Tọa độ: ${{poi.lat.toFixed(6)}}, ${{poi.lng.toFixed(6)}}</span>
      </div>
    `);
  }});
}}

renderPois(poisData);

function updateUI(data) {{
  document.querySelector('.val-done').textContent = data.done;
  document.querySelector('.val-discarded').textContent = data.discarded;
  document.querySelector('.val-queued').textContent = data.queued;
  document.querySelector('.val-pending').textContent = data.pending;
  document.querySelector('.val-poi').textContent = data.pois;

  document.querySelector('.progress-fill').style.width = data.pct.toFixed(2) + '%';
  document.querySelector('.progress-label').textContent = data.pct.toFixed(2) + '% hoàn thành';
}}

// Tu dong luu trang thai map khi keo, zoom
map.on('moveend', () => {{
  const center = map.getCenter();
  localStorage.setItem('map_lat', center.lat);
  localStorage.setItem('map_lng', center.lng);
}});
map.on('zoomend', () => {{
  localStorage.setItem('map_zoom', map.getZoom());
}});

// ── Auto-update mượt mà in-place qua map_data.json hoặc map_data.js ─────────────
(function() {{
  let lastTs = {page_ts};
  const DATA_URL = 'map_data.json';
  const dot = document.querySelector('.dot-live');
  const lbl = document.getElementById('live-status');

  function setLive(active) {{
    if (active) {{
      dot.classList.add('active');
      lbl.textContent = 'Tự động cập nhật (≲2s)';
    }} else {{
      dot.classList.remove('active');
      lbl.textContent = 'Không có kết nối';
    }}
  }}

  async function checkStatus() {{
    if (window.location.protocol === 'file:') {{
      // Dùng thẻ script động để bypass CORS bảo mật của file://
      const oldScript = document.getElementById('map-data-script');
      if (oldScript) {{
        oldScript.remove();
      }}
      const script = document.createElement('script');
      script.id = 'map-data-script';
      script.src = 'map_data.js?_=' + Date.now();
      script.onload = function() {{
        if (window.MAP_DATA) {{
          setLive(true);
          const data = window.MAP_DATA;
          if (data.ts !== lastTs) {{
            window.location.reload();
          }}
        }} else {{
          setLive(false);
        }}
      }};
      script.onerror = function() {{
        setLive(false);
      }};
      document.body.appendChild(script);
    }} else {{
      // Dùng fetch thông thường khi chạy qua HTTP Server
      try {{
        const resp = await fetch(DATA_URL + '?_=' + Date.now());
        if (!resp.ok) {{ setLive(false); return; }}
        const data = await resp.json();
        setLive(true);
        if (data.ts !== lastTs) {{
          window.location.reload();
        }}
      }} catch (e) {{
        setLive(false);
      }}
    }}
  }}

  // Kiểm tra mỗi 2 giây
  checkStatus();
  setInterval(checkStatus, 2000);
}})();
</script>
</body>
</html>
"""

_MAP_OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "map_viewer.html")


def build_and_save(
    all_tiles,
    visited: set,
    queued: set,
    pois: list,
    discarded: set = None,
    captured: set = None,
    out_path: str = _MAP_OUT,
) -> str:
    """
    Sinh map_viewer.html tu trang thai hien tai va ghi ra disk.
    Tra ve duong dan den file HTML.
    """
    if discarded is None:
        discarded = set()
    if captured is None:
        captured = set()
    total   = len(all_tiles)
    done    = len(visited)
    q_count = len(queued)
    disc    = len(discarded)
    pending = max(0, total - done - q_count)
    pct     = done / total * 100 if total else 0
    poi_count = len(pois)

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

    geojson_obj = build_geojson(all_tiles, visited, queued, discarded, captured)
    geojson_str = json.dumps(geojson_obj)
    display_zoom = max(10, config.ZOOM_LEVEL - 6)

    # Trích xuất dữ liệu POI nhẹ nhàng để nhúng trực tiếp vẽ lên bản đồ
    formatted_pois = []
    for p in pois:
        lat = p.get("approx_lat")
        lng = p.get("approx_lng")
        name = p.get("name", "Không rõ tên")
        if lat is not None and lng is not None:
            formatted_pois.append({
                "name": name,
                "lat": lat,
                "lng": lng,
                "type": p.get("type", "Địa điểm")
            })
    pois_json_str = json.dumps(formatted_pois, ensure_ascii=False)

    # Ghi map_data.json để map_viewer.html fetch động (tránh load lại trang gây trắng màn hình)
    import time
    current_ts = int(time.time() * 1000)
    map_data_path = os.path.join(os.path.dirname(out_path), "map_data.json")
    try:
        data = {
            "ts": current_ts,
            "total": total,
            "done": done,
            "queued": q_count,
            "discarded": disc,
            "pending": pending,
            "pois": poi_count,
            "pct": pct,
            "geojson": geojson_obj
        }
        tmp = map_data_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, map_data_path)

        # Ghi map_data.js để nhúng trực tiếp dạng script tag bypass CORS của file://
        map_js_path = os.path.join(os.path.dirname(out_path), "map_data.js")
        js_content = f"window.MAP_DATA = {json.dumps(data, ensure_ascii=False)};"
        tmp_js = map_js_path + ".tmp"
        with open(tmp_js, "w", encoding="utf-8") as f:
            f.write(js_content)
        os.replace(tmp_js, map_js_path)
    except Exception:
        pass

    html = HTML_TEMPLATE.format(
        total=total, done=done, queued=q_count, discarded=disc,
        pending=pending, pois=poi_count, pct=pct,
        center_lat=config.CENTER_LAT,
        center_lng=config.CENTER_LNG,
        map_zoom=display_zoom,
        target_district=config.TARGET_DISTRICT,
        geojson=geojson_str,
        boundary_geojson=boundary_geojson_str,
        page_ts=current_ts,
        pois_data=pois_json_str,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


def main():
    visited, queued, results, discarded, captured = load_state()

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
    print(f"  POIs   : {len(results)}")

    out_path = build_and_save(all_tiles, visited, queued, results, discarded, captured)
    print(f"\nMap saved: {out_path}")
    print("Opening in browser...")
    webbrowser.open(f"file:///{out_path.replace(os.sep, '/')}")


if __name__ == "__main__":
    main()

