# =============================================================
# worker.py — Playwright browser worker
#
# Mỗi Worker:
#   - Duy trì 1 instance Chromium headless
#   - Điều hướng đến từng tile trên OSM
#   - Chụp screenshot → gửi sang vision.py
#   - Báo kết quả cho Coordinator
#   - Tự restart browser mỗi BROWSER_RESTART_EVERY tile
# =============================================================

import asyncio
import logging
import os
from typing import List, Tuple

from playwright.async_api import Browser, BrowserContext, Page, Playwright

from config import (
    BROWSER_RESTART_EVERY,
    DELAY_BETWEEN_REQ,
    MAX_RETRIES,
    PAGE_LOAD_TIMEOUT,
    PAGE_SETTLE_MS,
    SAVE_SCREENSHOTS,
    SCREENSHOT_DIR,
    SCREENSHOT_H,
    SCREENSHOT_W,
    SCREENSHOT_ZOOM,
    ZOOM_LEVEL,
    HEADLESS,
)
from grid import tile_center, tile_bbox
from vision import extract_pois_from_screenshot

logger = logging.getLogger(__name__)

TileCoord = Tuple[int, int]

# URL template OSM
_OSM_URL = "https://www.openstreetmap.org/#map={zoom}/{lat}/{lng}"


class Worker:
    """
    Playwright-based tile processor.

    Sử dụng:
        async with async_playwright() as pw:
            worker = Worker(worker_id=0, coordinator=coord)
            await worker.run(pw)
    """

    def __init__(self, worker_id: int, coordinator) -> None:
        self.id = worker_id
        self.coord = coordinator

        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

        self._tile_count = 0          # tiles xử lý kể từ lần start/restart cuối
        self._total_processed = 0     # tổng tiles đã xử lý

    # ── Browser lifecycle ────────────────────────────────────

    async def _start_browser(self) -> None:
        """Khởi động (hoặc restart) Chromium instance."""
        assert self._playwright is not None, "Playwright is not initialized"
        await self._close_browser()

        self._browser = await self._playwright.chromium.launch(
            headless=HEADLESS,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-extensions",
                "--disable-background-networking",
                "--disable-default-apps",
            ],
        )
        self._context = await self._browser.new_context(
            viewport={"width": SCREENSHOT_W, "height": SCREENSHOT_H},
            user_agent=(
                "Mozilla/5.0 (compatible; OSM-Research-Bot/2.0; "
                "+https://github.com/DuyBaoPhan/grid-osm)"
            ),
        )
        self._page = await self._context.new_page()
        self._tile_count = 0
        logger.info("[Worker %d] Browser started.", self.id)

    async def _close_browser(self) -> None:
        """Đóng browser an toàn, bỏ qua lỗi."""
        for obj in [self._page, self._context, self._browser]:
            if obj is not None:
                try:
                    await obj.close()
                except Exception:
                    pass
        self._browser = self._context = self._page = None

    # ── Main loop ────────────────────────────────────────────

    async def run(self, playwright: Playwright) -> None:
        """Vòng lặp chính của worker — chạy đến khi queue rỗng."""
        self._playwright = playwright
        await self._start_browser()

        try:
            while True:
                tile = await self.coord.get_next_tile()

                if tile is None:
                    # Queue tạm thời trống — worker khác có thể đang xử lý tile cuối
                    # Retry nhiều lần với thời gian chờ tăng dần
                    # (Playwright thật cần 30-60s/tile, nên cần chờ đủ lâu)
                    gave_up = True
                    for retry in range(1, 6):  # thử tối đa 5 lần
                        wait_sec = retry * 3.0   # 3s, 6s, 9s, 12s, 15s
                        logger.debug(
                            "[Worker %d] Queue empty, retry %d/5 in %.0fs...",
                            self.id, retry, wait_sec,
                        )
                        await asyncio.sleep(wait_sec)
                        tile = await self.coord.get_next_tile()
                        if tile is not None:
                            gave_up = False
                            break
                    if gave_up:
                        logger.info("[Worker %d] Queue confirmed empty — shutting down.", self.id)
                        break

                # Restart browser định kỳ để giải phóng RAM
                if self._tile_count >= BROWSER_RESTART_EVERY:
                    logger.info(
                        "[Worker %d] Restarting browser after %d tiles…",
                        self.id, self._tile_count,
                    )
                    await self._start_browser()

                # In trang thai truoc khi xu ly tile
                stats = self.coord.stats
                logger.info(
                    "[%d/%d] Tile (%d,%d) | Queue: %d remaining | POIs: %d collected",
                    stats["tiles_done"] + 1,
                    stats["total_tiles"],
                    tile[0], tile[1],
                    stats["queue_size"],
                    stats["pois_found"],
                )

                await self._process_tile(tile)

                self._tile_count += 1
                self._total_processed += 1

                await asyncio.sleep(DELAY_BETWEEN_REQ)

        finally:
            await self._close_browser()
            logger.info(
                "[Worker %d] Done. Total tiles processed: %d",
                self.id, self._total_processed,
            )

    # ── Tile processor ───────────────────────────────────────

    async def _process_tile(self, tile: TileCoord) -> None:
        """
        Xử lý 1 tile:
          1. Điều hướng đến OSM URL
          2. Chụp screenshot
          3. Gọi vision → danh sách POI
          4. Báo kết quả cho coordinator
        Retry MAX_RETRIES lần nếu lỗi.
        """
        tx, ty = tile
        lat, lng = tile_center(tx, ty, ZOOM_LEVEL)
        bbox = tile_bbox(tx, ty, ZOOM_LEVEL)
        # Dung SCREENSHOT_ZOOM de browser hien thi to hon, thay ro ten dia diem
        url = _OSM_URL.format(zoom=SCREENSHOT_ZOOM, lat=round(lat, 6), lng=round(lng, 6))

        for attempt in range(1, MAX_RETRIES + 2):
            try:
                screenshot = await self._capture_screenshot(url, bbox)
                break
            except Exception as exc:
                if attempt <= MAX_RETRIES:
                    logger.warning(
                        "[Worker %d] Screenshot attempt %d/%d failed for tile (%d,%d): %s",
                        self.id, attempt, MAX_RETRIES, tx, ty, exc,
                    )
                    await asyncio.sleep(attempt * 2.0)
                    # Restart browser nếu page không phản hồi
                    try:
                        await self._start_browser()
                    except Exception:
                        pass
                else:
                    logger.error(
                        "[Worker %d] Tile (%d,%d) skipped after %d attempts.",
                        self.id, tx, ty, MAX_RETRIES + 1,
                    )
                    # Re-queue để thử lại trong phiên sau
                    await self.coord._queue.put(tile)
                    return

        # Lưu screenshot debug nếu cần
        if SAVE_SCREENSHOTS:
            await self._save_screenshot(screenshot, tx, ty)
        logger.info("  [1/2] Screenshot captured -> sending to LLM...")

        # Nhận diện POI và cờ báo ranh giới quận
        poi_names, outside_district = await extract_pois_from_screenshot(screenshot)
        logger.info("  [2/2] LLM done.")

        # Ghi lại tên POI để resolve tọa độ qua DOM browser
        # Thay vì dùng ước lượng x,y từ LLM, dùng JS trong browser để tìm
        # đúng vị trí pixel của nhãn chữ trên bản đồ và chuyển sang lat/lng.
        lat_min, lng_min, lat_max, lng_max = bbox
        poi_name_list = [item.get("name", "").strip() for item in poi_names if item.get("name", "").strip()]

        # Bước 2b: Lấy tọa độ chính xác từ browser (ưu tiên) hoặc fallback LLM x,y
        browser_coords = await self._resolve_poi_coords_via_browser(poi_name_list)
        logger.info("  [Geo] Browser DOM resolved %d/%d POI coordinates.",
                    len(browser_coords), len(poi_name_list))

        pois = []
        for item in poi_names:
            name = item.get("name", "").strip()
            if not name:
                continue

            if name in browser_coords:
                # Tọa độ lấy từ Leaflet DOM — chính xác tuyệt đối
                poi_lat = browser_coords[name]["lat"]
                poi_lng = browser_coords[name]["lng"]
            else:
                # Fallback: toán học tile bbox từ ước lượng LLM x,y
                x_pct = max(0.0, min(100.0, float(item.get("x", 50))))
                y_pct = max(0.0, min(100.0, float(item.get("y", 50))))
                poi_lat = lat_max - (y_pct / 100.0) * (lat_max - lat_min)
                poi_lng = lng_min + (x_pct / 100.0) * (lng_max - lng_min)
                logger.debug("  [Geo] Fallback visual coords for: %s", name)

            # Lọc nghiêm ngặt: Chỉ giữ POI có tọa độ thực sự nằm trong khung quét (bbox) của tile hiện tại.
            # Với tọa độ fallback, nó luôn nằm trong bbox vì x_pct, y_pct được chặn từ 0-100%.
            # Với tọa độ thực từ DOM browser, ta kiểm tra xem nó có thực sự nằm trong tile này không.
            # Cho phép một dung sai cực kỳ nhỏ (epsilon = 1e-6 độ, khoảng 10cm) để tránh sai số dấu phẩy động ở biên.
            eps = 1e-6
            in_lat = (lat_min - eps) <= poi_lat <= (lat_max + eps)
            in_lng = (lng_min - eps) <= poi_lng <= (lng_max + eps)

            if in_lat and in_lng:
                pois.append({
                    "name":       name,
                    "approx_lat": poi_lat,
                    "approx_lng": poi_lng,
                    "tile_x":     tx,
                    "tile_y":     ty,
                })
            else:
                logger.info(
                    "  [Geo] Bỏ qua POI ngoài ranh giới ô quét: %s (%.6f, %.6f) — thuộc ô khác",
                    name, poi_lat, poi_lng
                )

        # 8 hàng xóm
        neighbors: List[TileCoord] = [
            (tx + dx, ty + dy)
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            if not (dx == 0 and dy == 0)
        ]

        await self.coord.report_result(tile, pois, neighbors, outside_district)

        if pois:
            poi_list = ", ".join(p["name"] for p in pois[:5])
            suffix = f" (+{len(pois)-5} more)" if len(pois) > 5 else ""
            logger.info(
                "  => %d POI found: %s%s",
                len(pois), poi_list, suffix,
            )
        else:
            logger.info("  => No POI found at this tile")

    async def _resolve_poi_coords_via_browser(self, poi_names: List[str]) -> dict:
        """
        Dùng JavaScript trong trình duyệt đang mở để tìm phần tử DOM của từng
        nhãn POI trên bản đồ OSM (SVG <text>, Leaflet popup, icon label...),
        lấy vị trí pixel của nó rồi chuyển sang lat/lng chính xác tuyệt đối
        bằng Leaflet's containerPointToLatLng() — không cần bất kỳ API bên ngoài.

        Trả về dict: {"Tên POI": {"lat": ..., "lng": ...}}
        Các POI không tìm thấy trong DOM sẽ không có trong dict (sẽ fallback sang visual coords).
        """
        if not poi_names or self._page is None:
            return {}

        import json as _json
        results: dict = {}

        for name in poi_names:
            try:
                coords = await self._page.evaluate(
                    """
                    (searchText) => {
                        // Lấy Leaflet map instance
                        let mapInstance = null;
                        if (typeof OSM !== 'undefined' && OSM.map) {
                            mapInstance = OSM.map;
                        } else if (window.MAP) {
                            mapInstance = window.MAP;
                        } else if (window.map) {
                            mapInstance = window.map;
                        } else {
                            for (let key in window) {
                                try {
                                    if (window[key] && window[key]._layers
                                        && typeof window[key].invalidateSize === 'function') {
                                        mapInstance = window[key];
                                        break;
                                    }
                                } catch (e) {}
                            }
                        }
                        if (!mapInstance) return null;

                        const mapEl = document.getElementById('map') || document.body;

                        // 1. Tìm trong SVG <text> (nhãn đường phố và tên địa điểm của OSM)
                        const svgTexts = mapEl.querySelectorAll('text');
                        for (const el of svgTexts) {
                            const content = (el.textContent || '').trim();
                            if (content === searchText || content.includes(searchText) || searchText.includes(content) && content.length > 3) {
                                const rect = el.getBoundingClientRect();
                                if (rect.width === 0 && rect.height === 0) continue;
                                const cx = rect.left + rect.width / 2;
                                const cy = rect.top + rect.height / 2;
                                try {
                                    const ll = mapInstance.containerPointToLatLng([cx, cy]);
                                    return { lat: ll.lat, lng: ll.lng, source: 'svg-text' };
                                } catch(e) {}
                            }
                        }

                        // 2. Tìm trong Leaflet marker icons và tooltip labels
                        const labels = mapEl.querySelectorAll(
                            '.leaflet-marker-icon, .leaflet-tooltip, .leaflet-popup-content, [class*="label"]'
                        );
                        for (const el of labels) {
                            const content = (el.textContent || '').trim();
                            if (content === searchText || content.includes(searchText)) {
                                const rect = el.getBoundingClientRect();
                                if (rect.width === 0 && rect.height === 0) continue;
                                const cx = rect.left + rect.width / 2;
                                const cy = rect.top + rect.height / 2;
                                try {
                                    const ll = mapInstance.containerPointToLatLng([cx, cy]);
                                    return { lat: ll.lat, lng: ll.lng, source: 'leaflet-label' };
                                } catch(e) {}
                            }
                        }

                        // 3. Tìm rộng hơn trong toàn bộ DOM (fallback cho các renderer lạ)
                        const walker = document.createTreeWalker(
                            mapEl, NodeFilter.SHOW_TEXT, null
                        );
                        let node;
                        while ((node = walker.nextNode())) {
                            const content = (node.textContent || '').trim();
                            if (content === searchText) {
                                const el = node.parentElement;
                                if (!el) continue;
                                const rect = el.getBoundingClientRect();
                                if (rect.width === 0 && rect.height === 0) continue;
                                const cx = rect.left + rect.width / 2;
                                const cy = rect.top + rect.height / 2;
                                try {
                                    const ll = mapInstance.containerPointToLatLng([cx, cy]);
                                    return { lat: ll.lat, lng: ll.lng, source: 'text-node' };
                                } catch(e) {}
                            }
                        }

                        return null;
                    }
                    """,
                    name
                )
                if coords and "lat" in coords and "lng" in coords:
                    results[name] = coords
                    logger.debug(
                        "  [Geo] %-40s → (%.6f, %.6f) via %s",
                        name, coords["lat"], coords["lng"], coords.get("source", "?")
                    )
            except Exception as exc:
                logger.debug("  [Geo] DOM lookup failed for '%s': %s", name, exc)

        return results

    async def _capture_screenshot(self, url: str, bbox: Tuple[float, float, float, float]) -> bytes:
        """Điều hướng đến URL và chụp screenshot."""
        assert self._page is not None, "Page is not initialized"
        await self._page.goto(
            url,
            wait_until="networkidle",
            timeout=PAGE_LOAD_TIMEOUT,
        )
        
        # 1. Inject CSS ẩn toàn bộ UI rác (welcome panel, header, banner) và cho bản đồ phóng full màn hình
        css_hide_clutter = """
        #header, #sidebar, .welcome, .banner, #banner, .announcement, .flash-wrap, #flash, .cookie-consent, #cookie-consent {
            display: none !important;
        }
        #map {
            left: 0 !important;
            top: 0 !important;
            width: 100% !important;
            height: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
            position: absolute !important;
        }
        """
        try:
            await self._page.add_style_tag(content=css_hide_clutter)
        except Exception as exc:
            logger.debug("Could not hide OSM UI elements: %s", exc)

        # 2. Chạy Javascript xóa hoàn toàn các phần tử rác khỏi DOM, vẽ khung quét màu Neon Blue và tính toán ô cắt (crop_box)
        crop_box = None
        try:
            crop_box = await self._page.evaluate(f"""() => {{
                const selectors = [
                    '#header', '#sidebar', '.welcome', '#banner', '.banner', 
                    '.announcement', '.flash-wrap', '#flash', '.cookie-consent'
                ];
                selectors.forEach(sel => {{
                    document.querySelectorAll(sel).forEach(el => el.remove());
                }});
                
                // Ép bản đồ fill toàn màn hình
                const mapEl = document.getElementById('map');
                if (mapEl) {{
                    mapEl.style.setProperty('left', '0px', 'important');
                    mapEl.style.setProperty('top', '0px', 'important');
                    mapEl.style.setProperty('width', '100%', 'important');
                    mapEl.style.setProperty('height', '100%', 'important');
                    mapEl.style.setProperty('position', 'absolute', 'important');
                    mapEl.style.setProperty('margin', '0px', 'important');
                    mapEl.style.setProperty('padding', '0px', 'important');
                }}
                
                // Kích hoạt cập nhật kích thước Leaflet map và vẽ khung quét
                let mapInstance = null;
                if (typeof OSM !== 'undefined' && OSM.map) {{
                    mapInstance = OSM.map;
                }} else if (window.MAP) {{
                    mapInstance = window.MAP;
                }} else if (window.map) {{
                    mapInstance = window.map;
                }} else {{
                    for (let key in window) {{
                        try {{
                            if (window[key] && window[key]._layers && typeof window[key].invalidateSize === 'function') {{
                                mapInstance = window[key];
                                break;
                            }}
                        }} catch (e) {{}}
                    }}
                }}

                let box = null;
                if (mapInstance) {{
                    mapInstance.invalidateSize();
                    
                    // Vẽ khung quét Neon Blue thể hiện phạm vi chụp ảnh của worker
                    const bounds = [[{bbox[0]}, {bbox[1]}], [{bbox[2]}, {bbox[3]}]];
                    if (window.L) {{
                        if (window.activeWorkerBbox) {{
                            window.activeWorkerBbox.remove();
                        }}
                        window.activeWorkerBbox = window.L.rectangle(bounds, {{
                            color: '#00d2ff',
                            weight: 3,
                            fillColor: '#00d2ff',
                            fillOpacity: 0.08,
                            dashArray: '8, 8',
                            interactive: false
                        }}).addTo(mapInstance);

                        // Tính toán tọa độ pixel của bounding box trên màn hình
                        try {{
                            const p1 = mapInstance.latLngToContainerPoint(window.L.latLng({bbox[2]}, {bbox[1]})); // Top-Left (lat_max, lng_min)
                            const p2 = mapInstance.latLngToContainerPoint(window.L.latLng({bbox[0]}, {bbox[3]})); // Bottom-Right (lat_min, lng_max)
                            box = {{
                                x: Math.round(p1.x),
                                y: Math.round(p1.y),
                                width: Math.round(p2.x - p1.x),
                                height: Math.round(p2.y - p1.y)
                            }};
                        }} catch(err) {{}}
                    }}
                }}
                return box;
            }}""")
        except Exception as exc:
            logger.debug("Could not remove OSM elements or draw scan box via JS: %s", exc)

        await self._page.wait_for_timeout(PAGE_SETTLE_MS)
        screenshot_bytes = await self._page.screenshot(type="png")

        # 3. Sử dụng Pillow để crop ảnh theo tọa độ ô lưới cộng thêm padding 128px để giữ trọn vẹn nhãn chữ ở biên ô lưới
        if crop_box and all(k in crop_box for k in ["x", "y", "width", "height"]):
            try:
                from PIL import Image
                import io
                
                img = Image.open(io.BytesIO(screenshot_bytes))
                w, h = img.size
                
                padding = 128
                x1 = max(0, int(crop_box["x"]) - padding)
                y1 = max(0, int(crop_box["y"]) - padding)
                x2 = min(w, int(crop_box["x"]) + int(crop_box["width"]) + padding)
                y2 = min(h, int(crop_box["y"]) + int(crop_box["height"]) + padding)
                
                # Chỉ crop nếu kích thước hợp lệ
                if x2 > x1 and y2 > y1:
                    cropped_img = img.crop((x1, y1, x2, y2))
                    output_bytes = io.BytesIO()
                    cropped_img.save(output_bytes, format="PNG")
                    logger.info("  [Crop] Screenshot cropped to tile bounds with %dpx padding: %dx%d px", padding, x2-x1, y2-y1)
                    return output_bytes.getvalue()
            except Exception as crop_err:
                logger.warning("Could not crop screenshot: %s", crop_err)

        return screenshot_bytes

    async def _save_screenshot(self, data: bytes, tx: int, ty: int) -> None:
        """Lưu screenshot ra disk (chỉ dùng khi debug)."""
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        path = os.path.join(SCREENSHOT_DIR, f"tile_{tx}_{ty}.png")
        try:
            with open(path, "wb") as f:
                f.write(data)
        except Exception as exc:
            logger.debug("Could not save screenshot: %s", exc)
