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
from typing import List, Tuple, Optional

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
            device_scale_factor=2,  # Kích hoạt chế độ High DPI giúp chữ siêu sắc nét
            user_agent=(
                "Mozilla/5.0 (compatible; OSM-Research-Bot/2.0; "
                "+https://github.com/DuyBaoPhan/grid-osm)"
            ),
            bypass_csp=True,
        )
        self._page = await self._context.new_page()

        # Inject CSS to hide all clutter elements (welcome panel, header, banners) before they render!
        await self._page.add_init_script("""
            const style = document.createElement('style');
            style.textContent = `
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
            `;
            document.documentElement.appendChild(style);
        """)

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
          1. Khởi động browser
          2. Điều hướng đến OSM URL
          3. Chụp screenshot và trích xuất TOÀN BỘ tọa độ DOM địa điểm cùng một lúc
          4. Đóng browser ngay lập tức để tiết kiệm RAM, CPU và dọn dẹp màn hình!
          5. Gọi vision → danh sách POI (chạy ngầm 30-60s không cần browser)
          6. Báo kết quả cho coordinator
        Retry MAX_RETRIES lần nếu lỗi.
        """
        tx, ty = tile
        lat, lng = tile_center(tx, ty, ZOOM_LEVEL)
        bbox = tile_bbox(tx, ty, ZOOM_LEVEL)
        url = _OSM_URL.format(zoom=SCREENSHOT_ZOOM, lat=round(lat, 6), lng=round(lng, 6))

        # Đảm bảo khởi động trình duyệt cho ô quét này
        await self._start_browser()

        browser_coords = {}
        for attempt in range(1, MAX_RETRIES + 2):
            try:
                screenshot = await self._capture_screenshot(url, bbox)
                # Trích xuất toàn bộ nhãn và tọa độ hiển thị trong DOM hiện tại
                browser_coords = await self._extract_all_visible_poi_coords_from_browser()
                # Báo cáo ngay cho coordinator rằng đã chụp ảnh xong để vẽ ô màu xanh neon blue lên bản đồ!
                await self.coord.report_captured(tile)
                break
            except Exception as exc:
                if attempt <= MAX_RETRIES:
                    logger.warning(
                        "[Worker %d] Screenshot attempt %d/%d failed for tile (%d,%d): %s",
                        self.id, attempt, MAX_RETRIES, tx, ty, exc,
                    )
                    await asyncio.sleep(attempt * 2.0)
                    try:
                        await self._start_browser()
                    except Exception:
                        pass
                else:
                    logger.error(
                        "[Worker %d] Tile (%d,%d) skipped after %d attempts.",
                        self.id, tx, ty, MAX_RETRIES + 1,
                    )
                    await self._close_browser()
                    # Re-queue để thử lại trong phiên sau
                    await self.coord._queue.put(tile)
                    return

        # Lưu screenshot debug nếu cần
        if SAVE_SCREENSHOTS:
            await self._save_screenshot(screenshot, tx, ty)
        logger.info("  [1/2] Screenshot captured -> sending to LLM...")

        # Nhận diện POI và cờ báo ranh giới quận
        poi_names, outside_district = await extract_pois_from_screenshot(
            screenshot, bbox, tx=tx, ty=ty, zoom=SCREENSHOT_ZOOM
        )
        logger.info("  [2/2] LLM done.")

        # Đóng trình duyệt sau khi LLM đã đọc xong và xuất ra thông tin địa điểm của ô quét này!
        await self._close_browser()

        # Lọc và khớp tọa độ địa điểm
        lat_min, lng_min, lat_max, lng_max = bbox
        
        # Hàm so khớp mềm tên POI từ LLM với nhãn DOM trích xuất được
        def find_dom_match(poi_name: str, dom_coords: dict) -> Optional[dict]:
            # 1. Khớp chính xác tuyệt đối
            if poi_name in dom_coords:
                return dom_coords[poi_name]
            # 2. Khớp không phân biệt chữ hoa thường
            poi_name_lower = poi_name.lower()
            for k, v in dom_coords.items():
                if k.lower() == poi_name_lower:
                    return v
            # 3. Khớp một phần (substring)
            for k, v in dom_coords.items():
                k_lower = k.lower()
                if len(k) > 3 and (k_lower in poi_name_lower or poi_name_lower in k_lower):
                    return v
            return None

        pois = []
        for item in poi_names:
            name = item.get("name", "").strip()
            if not name:
                continue

            dom_match = find_dom_match(name, browser_coords)
            if dom_match:
                # Tọa độ lấy từ Leaflet DOM — chính xác tuyệt đối
                poi_lat = dom_match["lat"]
                poi_lng = dom_match["lng"]
            else:
                # Giải quyết tọa độ từ nhãn ô lưới A1 -> C3 (GPS Grid Cell Resolver)
                cell = str(item.get("cell", "B2")).strip().upper()
                side = str(item.get("side", "center")).strip().lower()
                r_idx, c_idx = 1, 1  # Mặc định là ô trung tâm B2
                if len(cell) == 2:
                    r_char, c_char = cell[0], cell[1]
                    if r_char in "ABC" and c_char in "123":
                        r_idx = "ABC".index(r_char)
                        c_idx = "123".index(c_char)

                # Sub-cell offset dựa trên vị trí (side) trong ô:
                # sx, sy ∈ [0.0, 1.0] — tỉ lệ trong phạm vi của ô 1/3 đó
                _SIDE_OFFSETS = {
                    "top-left":     (0.2, 0.2),
                    "top-center":   (0.5, 0.2),
                    "top-right":    (0.8, 0.2),
                    "left-center":  (0.2, 0.5),
                    "center":       (0.5, 0.5),
                    "right-center": (0.8, 0.5),
                    "bottom-left":  (0.2, 0.8),
                    "bottom-center":(0.5, 0.8),
                    "bottom-right": (0.8, 0.8),
                }
                sx, sy = _SIDE_OFFSETS.get(side, (0.5, 0.5))

                # x_pct, y_pct = vị trí % trong toàn bộ ảnh đã crop
                x_pct = (c_idx + sx) * (100.0 / 3.0)
                y_pct = (r_idx + sy) * (100.0 / 3.0)

                # Chiều cao và chiều rộng của ô tile tiêu chuẩn là 256px
                # Ảnh gửi cho LLM được crop rộng hơn: thêm padding 25px ở cả 4 cạnh (tổng kích thước 306x306px)
                padding_lat = (25.0 / 256.0) * (lat_max - lat_min)
                padding_lng = (25.0 / 256.0) * (lng_max - lng_min)

                cropped_lat_max = lat_max + padding_lat
                cropped_lat_min = lat_min - padding_lat
                cropped_lng_min = lng_min - padding_lng
                cropped_lng_max = lng_max + padding_lng

                poi_lat = cropped_lat_max - (y_pct / 100.0) * (cropped_lat_max - cropped_lat_min)
                poi_lng = cropped_lng_min + (x_pct / 100.0) * (cropped_lng_max - cropped_lng_min)
                logger.info(
                    "  [Geo] Resolved '%s' → cell=%s side=%s → (%.6f, %.6f) [x:%.1f%% y:%.1f%%]",
                    name, cell, side, poi_lat, poi_lng, x_pct, y_pct
                )

            # Lọc nghiêm ngặt: Chỉ giữ POI có tọa độ thực sự nằm trong khung quét (bbox) của tile hiện tại.
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

    async def _extract_all_visible_poi_coords_from_browser(self) -> dict:
        """
        Trích xuất TOÀN BỘ nhãn địa điểm hiển thị trên bản đồ DOM hiện tại
        và tính toán tọa độ lat/lng chính xác của chúng bằng containerPointToLatLng.
        Trả về dict: {poi_name: {lat, lng}}
        """
        if self._page is None:
            return {}

        try:
            coords = await self._page.evaluate(
                """
                () => {
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
                    if (!mapInstance) return {};

                    const mapEl = document.getElementById('map') || document.body;
                    const results = {};

                    // 1. Quét SVG text
                    const svgTexts = mapEl.querySelectorAll('text');
                    for (const el of svgTexts) {
                        const content = (el.textContent || '').trim();
                        if (content.length > 2) {
                            const rect = el.getBoundingClientRect();
                            if (rect.width === 0 && rect.height === 0) continue;
                            const cx = rect.left + rect.width / 2;
                            const cy = rect.top + rect.height / 2;
                            try {
                                const ll = mapInstance.containerPointToLatLng([cx, cy]);
                                results[content] = { lat: ll.lat, lng: ll.lng, source: 'svg-text' };
                            } catch(e) {}
                        }
                    }

                    // 2. Quét Leaflet markers/tooltips
                    const labels = mapEl.querySelectorAll(
                        '.leaflet-marker-icon, .leaflet-tooltip, .leaflet-popup-content, [class*="label"]'
                    );
                    for (const el of labels) {
                        const content = (el.textContent || '').trim();
                        if (content.length > 2) {
                            const rect = el.getBoundingClientRect();
                            if (rect.width === 0 && rect.height === 0) continue;
                            const cx = rect.left + rect.width / 2;
                            const cy = rect.top + rect.height / 2;
                            try {
                                const ll = mapInstance.containerPointToLatLng([cx, cy]);
                                results[content] = { lat: ll.lat, lng: ll.lng, source: 'leaflet-label' };
                            } catch(e) {}
                        }
                    }

                    return results;
                }
                """
            )
            return coords if isinstance(coords, dict) else {}
        except Exception as exc:
            logger.warning("[Worker %d] Failed to extract DOM coords: %s", self.id, exc)
            return {}

    async def _capture_screenshot(self, url: str, bbox: Tuple[float, float, float, float]) -> bytes:
        """Điều hướng đến URL và chụp screenshot."""
        assert self._page is not None, "Page is not initialized"
        await self._page.goto(
            url,
            wait_until="domcontentloaded",
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

        # 2. Chạy Javascript xóa hoàn toàn các phần tử rác khỏi DOM, vẽ khung quét màu Neon Blue bằng DOM và tính toán ô cắt (crop_box)
        crop_box = None
        try:
            crop_box = await self._page.evaluate(f"""async () => {{
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
                
                // Chờ và kích hoạt cập nhật kích thước Leaflet map
                const waitAndCenter = () => {{
                    return new Promise((resolve) => {{
                        let attempts = 0;
                        const interval = setInterval(() => {{
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
                            if (mapInstance) {{
                                clearInterval(interval);
                                mapInstance.setView([{lat}, {lng}], {SCREENSHOT_ZOOM}, {{ animate: false }});
                                mapInstance.invalidateSize();
                                resolve(true);
                            }} else {{
                                attempts++;
                                if (attempts > 100) {{ // 5 seconds timeout
                                    clearInterval(interval);
                                    resolve(false);
                                }}
                            }}
                        }}, 50);
                    }});
                }};
                
                await waitAndCenter();

                // Vẽ khung quét Neon Blue bằng DOM Injection (đảm bảo hiển thị 100% đáng tin cậy trên mọi giao diện)
                let scanBox = document.getElementById('active-worker-scan-box');
                if (!scanBox) {{
                    scanBox = document.createElement('div');
                    scanBox.id = 'active-worker-scan-box';
                    document.body.appendChild(scanBox);
                }}
                scanBox.style.position = 'fixed';
                scanBox.style.left = '384px';
                scanBox.style.top = '256px';
                scanBox.style.width = '256px';
                scanBox.style.height = '256px';
                scanBox.style.border = '3px dashed #00d2ff';
                scanBox.style.backgroundColor = 'rgba(0, 210, 255, 0.08)';
                scanBox.style.pointerEvents = 'none';
                scanBox.style.zIndex = '99999';

                // Trả về tọa độ pixel hình học chuẩn xác của ô tile 256x256 ở giữa màn hình 1024x768
                return {{
                    x: 384,
                    y: 256,
                    width: 256,
                    height: 256
                }};
            }}""")
        except Exception as exc:
            logger.debug("Could not remove OSM elements or draw scan box via JS: %s", exc)

        await self._page.wait_for_timeout(PAGE_SETTLE_MS)
        screenshot_bytes = await self._page.screenshot(type="png")

        # 3. Sử dụng Pillow để crop và nén ảnh dưới dạng JPEG chất lượng cao cùng các kỹ thuật nâng cao chất lượng nhận diện!
        try:
            from PIL import Image, ImageEnhance, ImageFilter
            import io
            
            img = Image.open(io.BytesIO(screenshot_bytes))
            w, h = img.size
            
            # Tỷ lệ scale giữa pixel thực tế (physical pixels) và tọa độ CSS (logical pixels)
            scale = w / SCREENSHOT_W
            
            # Thêm border padding (khoảng đệm biên) để lấy trọn vẹn nhãn chữ ở sát rìa ô quét
            padding = 25
            padding_scaled = int(padding * scale)
            use_fallback = True
            
            if crop_box and all(k in crop_box for k in ["x", "y", "width", "height"]):
                try:
                    # Nhân tọa độ CSS từ DOM với tỷ lệ scale để chuyển đổi sang tọa độ pixel thực tế của màn hình High-DPI
                    x1 = max(0, int(crop_box["x"] * scale) - padding_scaled)
                    y1 = max(0, int(crop_box["y"] * scale) - padding_scaled)
                    x2 = min(w, int((crop_box["x"] + crop_box["width"]) * scale) + padding_scaled)
                    y2 = min(h, int((crop_box["y"] + crop_box["height"]) * scale) + padding_scaled)
                    if x2 > x1 and y2 > y1:
                        use_fallback = False
                except Exception:
                    pass
            
            if use_fallback:
                # Tính toán hình học cố định (Do OSM tự động căn giữa ô tile ở tâm màn hình)
                cx, cy = w / 2, h / 2
                tile_w = 256 * (w / 1024)
                tile_h = 256 * (h / 768)
                pad_w = padding * (w / 1024)
                pad_h = padding * (h / 768)
                
                x1 = max(0, int(cx - tile_w / 2 - pad_w))
                y1 = max(0, int(cy - tile_h / 2 - pad_h))
                x2 = min(w, int(cx + tile_w / 2 + pad_w))
                y2 = min(h, int(cy + tile_h / 2 + pad_h))
            
            # Thực hiện crop và áp dụng bộ lọc nâng cao chất lượng chữ
            if x2 > x1 and y2 > y1:
                cropped_img = img.crop((x1, y1, x2, y2))
                
                # Áp dụng bộ lọc làm sắc nét chữ và tăng tương phản giúp OCR đọc tốt hơn
                try:
                    cropped_img = cropped_img.filter(ImageFilter.SHARPEN)
                    enhancer = ImageEnhance.Contrast(cropped_img)
                    cropped_img = enhancer.enhance(1.25)
                except Exception as enh_err:
                    logger.debug("Image enhancement failed: %s", enh_err)

                # Vẽ lưới 3x3 ảo trực tiếp lên ảnh để hỗ trợ LLM định vị không gian chính xác tuyệt đối
                try:
                    from PIL import ImageDraw, ImageFont
                    # Chuyển sang RGBA để vẽ bán trong suốt
                    cropped_img = cropped_img.convert("RGBA")
                    draw = ImageDraw.Draw(cropped_img)
                    cw, ch = cropped_img.size
                    
                    # Vẽ lưới màu xanh neon dày và rõ nét hơn
                    grid_color = (0, 210, 255, 180) # RGBA với độ mờ cao
                    for i in range(1, 3):
                        x = int(cw * i / 3)
                        draw.line([(x, 0), (x, ch)], fill=grid_color, width=3)
                    for i in range(1, 3):
                        y = int(ch * i / 3)
                        draw.line([(0, y), (cw, y)], fill=grid_color, width=3)
                        
                    # Vẽ nhãn A1 -> C3 vào góc các ô lưới để LLM định vị cực kỳ dễ dàng
                    cells = [
                        ("A1", 0, 0), ("A2", 1, 0), ("A3", 2, 0),
                        ("B1", 0, 1), ("B2", 1, 1), ("B3", 2, 1),
                        ("C1", 0, 2), ("C2", 1, 2), ("C3", 2, 2)
                    ]
                    
                    # Tải font chữ rõ nét
                    try:
                        font = ImageFont.truetype("arial.ttf", 20)
                    except Exception:
                        try:
                            font = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 20)
                        except Exception:
                            font = ImageFont.load_default()

                    for label, col, row in cells:
                        lx = int(cw * col / 3) + 6
                        ly = int(ch * row / 3) + 6
                        
                        try:
                            l, t, r, b = draw.textbbox((lx, ly), label, font=font)
                            rect_box = [l - 4, t - 2, r + 4, b + 2]
                        except AttributeError:
                            if hasattr(draw, "textsize"):
                                w_t, h_t = draw.textsize(label, font=font)
                            else:
                                w_t, h_t = 16, 12
                            rect_box = [lx - 4, ly - 2, lx + w_t + 4, ly + h_t + 2]
                            
                        # Vẽ hình chữ nhật nền tối làm nổi bật chữ
                        draw.rectangle(rect_box, fill=(15, 23, 42, 220))
                        draw.text((lx, ly), label, fill=(255, 255, 255), font=font)
                except Exception as draw_err:
                    logger.debug("Could not draw grid overlays on screenshot: %s", draw_err)
                
                rgb_img = cropped_img.convert("RGB")
                output_bytes = io.BytesIO()
                # Lưu JPEG chất lượng cao 95% để giảm tối đa nhiễu nén làm hỏng viền chữ
                rgb_img.save(output_bytes, format="JPEG", quality=95)
                method_str = "Leaflet DOM" if not use_fallback else "Viewport Center Fallback"
                logger.info(
                    "  [Crop & Compress] Screenshot cropped via %s to %dx%d px (High-DPI 2x, quality=95%%) and enhanced with SHARPEN+Contrast",
                    method_str, x2 - x1, y2 - y1
                )
                return output_bytes.getvalue()
        except Exception as crop_err:
            logger.warning("Could not crop or compress screenshot: %s", crop_err)

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
