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
from grid import tile_center, tile_bbox, tile_viewport_bbox, pixel_to_gps
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
                "--start-maximized",
            ],
        )
        self._context = await self._browser.new_context(
            no_viewport=True,
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
                header, .header, .header-main, #header, .sidebar, #sidebar, .welcome, .banner, #banner, .announcement, .flash-wrap, #flash, .cookie-consent, #cookie-consent, .leaflet-control-container, #top-bar, .top-bar, #navbar, .navbar {
                    display: none !important;
                }
                #map {
                    left: 0 !important;
                    top: 0 !important;
                    width: 100vw !important;
                    height: 100vh !important;
                    margin: 0 !important;
                    padding: 0 !important;
                    position: fixed !important;
                    z-index: 999999 !important;
                }
                body, html {
                    overflow: hidden !important;
                    margin: 0 !important;
                    padding: 0 !important;
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
        bbox = tile_viewport_bbox(tx, ty, ZOOM_LEVEL)
        url = _OSM_URL.format(zoom=SCREENSHOT_ZOOM, lat=round(lat, 6), lng=round(lng, 6))

        # Đảm bảo khởi động trình duyệt cho ô quét này
        await self._start_browser()

        browser_coords = {}
        img_metadata = {}
        for attempt in range(1, MAX_RETRIES + 2):
            try:
                raw_screenshot, compressed_screenshot, img_metadata = await self._capture_screenshot(url, bbox)
                # Override bbox bằng góc thực tế của ảnh (khớp đúng vùng hiển thị, không phụ thuộc SCREENSHOT_H cứng)
                if img_metadata:
                    bbox = (
                        min(img_metadata["bottom_left_lat"], img_metadata["bottom_right_lat"]),
                        min(img_metadata["top_left_lng"],    img_metadata["bottom_left_lng"]),
                        max(img_metadata["top_left_lat"],    img_metadata["top_right_lat"]),
                        max(img_metadata["top_right_lng"],   img_metadata["bottom_right_lng"]),
                    )
                # Trích xuất toàn bộ nhãn và tọa độ hiển thị trong DOM hiện tại
                browser_coords = await self._extract_all_visible_poi_coords_from_browser()
                # Báo cáo ngay cho coordinator rằng đã chụp ảnh xong để vẽ ô màu xanh neon blue lên bản đồ!
                await self.coord.report_captured(tile, bbox)
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

        # Lưu screenshot gốc (không có lưới, không bị giảm chất lượng) nếu cần debug
        if SAVE_SCREENSHOTS:
            await self._save_screenshot(raw_screenshot, tx, ty)
        logger.info("  [1/2] Screenshot captured -> sending to LLM...")

        # Nhận diện POI và cờ báo ranh giới quận (gửi ảnh gốc sạch sẽ, không dùng lưới cho LLM)
        poi_names, outside_district = await extract_pois_from_screenshot(
            compressed_screenshot, bbox, tx=tx, ty=ty, zoom=SCREENSHOT_ZOOM, img_metadata=img_metadata
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
        pois = []
        for item in poi_names:
            name = item.get("name", "").strip()
            if not name:
                continue

            dom_match = find_dom_match(name, browser_coords)
            
            try:
                import math
                img_w = img_metadata.get("width", 612)
                img_h = img_metadata.get("height", 612)
                center_x = img_metadata.get("center_x", img_w / 2.0)
                center_y = img_metadata.get("center_y", img_h / 2.0)
                scale = img_metadata.get("scale", 2.0)
                crop_x1 = img_metadata.get("crop_x1", 0.0)
                crop_y1 = img_metadata.get("crop_y1", 0.0)

                # 1. Vision: Lấy tọa độ x, y của icon
                if dom_match:
                    # Nếu khớp DOM, quy đổi từ vị trí CSS trong DOM (cx, cy) sang tọa độ pixel thực trên ảnh crop
                    cx = dom_match.get("cx", center_x / scale)
                    cy = dom_match.get("cy", center_y / scale)
                    x_val = cx * scale - crop_x1
                    y_val = cy * scale - crop_y1
                    poi_lat = dom_match["lat"]
                    poi_lng = dom_match["lng"]
                else:
                    x_val = float(item.get("x", center_x))
                    y_val = float(item.get("y", center_y))

                # 2. Tính khoảng cách pixel vật lý (distance_pixels) từ tâm
                distance_pixels = math.sqrt((x_val - center_x)**2 + (y_val - center_y)**2)

                # 3. Tính bearing từ tâm theo pixel
                dx = x_val - center_x
                dy = center_y - y_val  # Trục Oy hướng lên (Bắc) là dương, pixel y đi xuống
                bearing_rad = math.atan2(dx, dy)
                bearing_deg = (math.degrees(bearing_rad) + 360.0) % 360.0

                # 4. Quy đổi pixel ➔ GPS
                # a. Quy đổi sang CSS pixels
                dist_css = distance_pixels / scale
                # b. Số mét trên mỗi CSS pixel ở vĩ độ hiện tại
                R_earth = 6378137.0
                tile_width_meters = (2 * math.pi * R_earth * math.cos(math.radians(lat))) / (2 ** SCREENSHOT_ZOOM)
                meters_per_css_pixel = tile_width_meters / 256.0
                # c. Tính khoảng cách mét
                distance_meters = dist_css * meters_per_css_pixel

                if not dom_match:
                    # Quy đổi kích thước ảnh và vị trí pixel từ vật lý sang CSS pixels trước khi tính toán
                    width_css = img_w / scale
                    height_css = img_h / scale
                    pixel_x_css = x_val / scale
                    pixel_y_css = y_val / scale

                    # Áp dụng công thức pixel_to_gps chính xác tiêu chuẩn Web Mercator
                    poi_lat, poi_lng = pixel_to_gps(
                        center_lat=lat,
                        center_lon=lng,
                        zoom=SCREENSHOT_ZOOM,
                        width=width_css,
                        height=height_css,
                        pixel_x=pixel_x_css,
                        pixel_y=pixel_y_css,
                        tile_size=256
                    )
                    
                    logger.info(
                        "  [GeoPixel] Resolved '%s' via exact Web Mercator pixel_to_gps → x=%d y=%d dist_px=%.1f dist_m=%.1fm bearing=%.1fdeg → (%.6f, %.6f)",
                        name, int(x_val), int(y_val), distance_pixels, distance_meters, bearing_deg, poi_lat, poi_lng
                    )
                else:
                    logger.info(
                        "  [LeafletDOM] Resolved '%s' via exact DOM matching → x=%d y=%d dist_px=%.1f dist_m=%.1fm bearing=%.1fdeg → (%.6f, %.6f)",
                        name, int(x_val), int(y_val), distance_pixels, distance_meters, bearing_deg, poi_lat, poi_lng
                    )

            except Exception as geo_err:
                logger.warning("  [GeoPixel] Lỗi khi tính toán tọa độ và khoảng cách: %s. Trở về center GPS mặc định.", geo_err)
                poi_lat = lat
                poi_lng = lng
                distance_pixels = 0.0
                distance_meters = 0.0
                bearing_deg = 0.0

            # Lọc nghiêm ngặt: Chỉ giữ POI có tọa độ thực sự nằm trong khung quét (bbox) của tile hiện tại.
            eps = 1e-6
            in_lat = (lat_min - eps) <= poi_lat <= (lat_max + eps)
            in_lng = (lng_min - eps) <= poi_lng <= (lng_max + eps)

            if in_lat and in_lng:
                pois.append({
                    "name":             name,
                    "approx_lat":       poi_lat,
                    "approx_lng":       poi_lng,
                    "tile_x":           tx,
                    "tile_y":           ty,
                    "distance_pixels":  round(distance_pixels, 1),
                    "distance_meters":  round(distance_meters, 1),
                    "bearing_degrees":  round(bearing_deg, 1),
                    "image_corners": {
                        "top_left":     {"lat": round(img_metadata.get("top_left_lat", 0.0), 6), "lng": round(img_metadata.get("top_left_lng", 0.0), 6)},
                        "top_right":    {"lat": round(img_metadata.get("top_right_lat", 0.0), 6), "lng": round(img_metadata.get("top_right_lng", 0.0), 6)},
                        "bottom_left":  {"lat": round(img_metadata.get("bottom_left_lat", 0.0), 6), "lng": round(img_metadata.get("bottom_left_lng", 0.0), 6)},
                        "bottom_right": {"lat": round(img_metadata.get("bottom_right_lat", 0.0), 6), "lng": round(img_metadata.get("bottom_right_lng", 0.0), 6)},
                    }
                })
            else:
                logger.info(
                    "  [Geo] Bỏ qua POI ngoài ranh giới ô quét: %s (%.6f, %.6f) — thuộc ô khác",
                    name, poi_lat, poi_lng
                )

        # 8 hàng xóm trong lưới tọa độ custom
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
                                results[content] = { lat: ll.lat, lng: ll.lng, cx: cx, cy: cy, source: 'svg-text' };
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
                                results[content] = { lat: ll.lat, lng: ll.lng, cx: cx, cy: cy, source: 'leaflet-label' };
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

    def _create_grid_overlay(self, img_bytes: bytes) -> bytes:
        """
        Vẽ lưới tham chiếu 5×5 (A-E × 1-5) lên ảnh screenshot.
        Giúp LLM xác định vị trí icon POI qua tên ô lưới (VD: 'C2')
        thay vì phải đoán tọa độ pixel chính xác.
        """
        from PIL import Image, ImageDraw, ImageFont
        import io

        img = Image.open(io.BytesIO(img_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        draw = ImageDraw.Draw(img)
        w, h = img.size

        cols, rows = 5, 5
        cell_w = w / cols
        cell_h = h / rows
        col_labels = "ABCDE"
        line_color = (220, 50, 50)

        # Chọn font rõ ràng
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except Exception:
            try:
                font = ImageFont.load_default(size=16)
            except TypeError:
                font = ImageFont.load_default()

        # Vẽ đường kẻ lưới dọc
        for i in range(1, cols):
            x = int(i * cell_w)
            draw.line([(x, 0), (x, h)], fill=line_color, width=2)
        # Vẽ đường kẻ lưới ngang
        for j in range(1, rows):
            y = int(j * cell_h)
            draw.line([(0, y), (w, y)], fill=line_color, width=2)
        # Viền ngoài
        draw.rectangle([(0, 0), (w - 1, h - 1)], outline=line_color, width=2)

        # Vẽ nhãn ô ở góc trên-trái mỗi ô
        for i in range(cols):
            for j in range(rows):
                label = f"{col_labels[i]}{j + 1}"
                lx = int(i * cell_w + 4)
                ly = int(j * cell_h + 3)
                # Nền trắng cho dễ đọc
                try:
                    bbox_t = draw.textbbox((lx, ly), label, font=font)
                    draw.rectangle(
                        [bbox_t[0] - 2, bbox_t[1] - 1, bbox_t[2] + 2, bbox_t[3] + 1],
                        fill=(255, 255, 255),
                    )
                except Exception:
                    pass
                draw.text((lx, ly), label, fill=line_color, font=font)

        # Vẽ dấu + ở tâm ảnh để LLM dễ định hướng
        cx, cy = w // 2, h // 2
        cross_size = 8
        draw.line([(cx - cross_size, cy), (cx + cross_size, cy)], fill=line_color, width=1)
        draw.line([(cx, cy - cross_size), (cx, cy + cross_size)], fill=line_color, width=1)

        output = io.BytesIO()
        img.save(output, format="JPEG", quality=95)
        logger.info("  [GridOverlay] Đã vẽ lưới 5×5 lên ảnh %dx%d px", w, h)
        return output.getvalue()

    async def _capture_screenshot(self, url: str, bbox: Tuple[float, float, float, float]) -> Tuple[bytes, dict]:
        """Điều hướng đến URL và chụp screenshot và trả về ảnh cùng metadata kích thước."""
        lat = (bbox[0] + bbox[2]) / 2.0
        lng = (bbox[1] + bbox[3]) / 2.0
        assert self._page is not None, "Page is not initialized"
        # OSM uses hash-based URLs (#map=zoom/lat/lng) which can cause ERR_ABORTED
        # when Playwright fires domcontentloaded too early. Retry with 'load' on abort.
        try:
            await self._page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=PAGE_LOAD_TIMEOUT,
            )
        except Exception as nav_exc:
            nav_msg = str(nav_exc)
            if "ERR_ABORTED" in nav_msg or "frame was detached" in nav_msg or "net::" in nav_msg:
                logger.warning("  [Nav] goto ERR_ABORTED (hash URL quirk) — retrying with 'load': %s", nav_msg[:120])
                try:
                    await self._page.goto(
                        url,
                        wait_until="load",
                        timeout=PAGE_LOAD_TIMEOUT,
                    )
                except Exception as retry_exc:
                    retry_msg = str(retry_exc)
                    if "ERR_ABORTED" in retry_msg or "frame was detached" in retry_msg:
                        # ERR_ABORTED on hash navigation is non-fatal — page content still loaded
                        logger.debug("  [Nav] Second goto also aborted (non-fatal for hash URL): %s", retry_msg[:80])
                    else:
                        raise
            else:
                raise
        
        # 1. Inject CSS ẩn toàn bộ UI rác (welcome panel, header, banner) và cho bản đồ phóng full màn hình
        css_hide_clutter = """
        header, .header, .header-main, #header, .sidebar, #sidebar, .welcome, .banner, #banner, .announcement, .flash-wrap, #flash, .cookie-consent, #cookie-consent, .leaflet-control-container, #top-bar, .top-bar, #navbar, .navbar {
            display: none !important;
        }
        #map {
            left: 0 !important;
            top: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            margin: 0 !important;
            padding: 0 !important;
            position: fixed !important;
            z-index: 999999 !important;
        }
        body, html {
            overflow: hidden !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        """
        try:
            await self._page.add_style_tag(content=css_hide_clutter)
        except Exception as exc:
            logger.debug("Could not hide OSM UI elements: %s", exc)

        # 2. Chạy Javascript xóa hoàn toàn các phần tử rác khỏi DOM và tính toán ô cắt (crop_box)
        crop_box = None
        try:
            crop_box = await self._page.evaluate(f"""async () => {{
                const selectors = [
                    'header', '.header', '.header-main', '#header', '.sidebar', '#sidebar', '.welcome', '#banner', '.banner', 
                    '.announcement', '.flash-wrap', '#flash', '.cookie-consent', '.leaflet-control-container'
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
                    mapEl.style.setProperty('position', 'fixed', 'important');
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
                                mapInstance.invalidateSize({{ animate: false }});
                                mapInstance.setView([{lat}, {lng}], {SCREENSHOT_ZOOM}, {{ animate: false }});
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

                // Vẽ khung quét bao toàn viewport (đúng với những gì worker thấy)
                let scanBox = document.getElementById('active-worker-scan-box');
                if (!scanBox) {{
                    scanBox = document.createElement('div');
                    scanBox.id = 'active-worker-scan-box';
                    document.body.appendChild(scanBox);
                }}
                scanBox.style.position = 'fixed';
                scanBox.style.left = '0px';
                scanBox.style.top = '0px';
                scanBox.style.width = '100vw';
                scanBox.style.height = '100vh';
                scanBox.style.border = '2px solid rgba(0, 210, 255, 0.4)';
                scanBox.style.backgroundColor = 'transparent';
                scanBox.style.pointerEvents = 'none';
                scanBox.style.zIndex = '10000000';
                scanBox.style.boxSizing = 'border-box';

                // Trả về null — dùng full viewport, không crop
                return null;
            }}""")
        except Exception as exc:
            logger.debug("Could not remove OSM elements or draw scan box via JS: %s", exc)

        try:
            await self._page.wait_for_load_state("networkidle", timeout=4000)
        except Exception:
            pass
        await self._page.wait_for_timeout(PAGE_SETTLE_MS)
        
        # Get actual viewport size dynamically via Javascript since page.viewport_size is None when no_viewport=True
        w_viewport, h_viewport = SCREENSHOT_W, SCREENSHOT_H
        try:
            v_size = await self._page.evaluate("() => ({ width: window.innerWidth, height: window.innerHeight })")
            if v_size and "width" in v_size and "height" in v_size:
                w_viewport = int(v_size["width"])
                h_viewport = int(v_size["height"])
        except Exception as eval_exc:
            logger.debug("Failed to evaluate viewport size via JS: %s", eval_exc)

        screenshot_bytes = await self._page.screenshot(type="png")

        # 3. Nén ảnh full viewport sang JPEG chất lượng cao — không crop, LLM thấy đúng như worker
        try:
            from PIL import Image, ImageEnhance, ImageFilter
            import io

            img = Image.open(io.BytesIO(screenshot_bytes))
            w_orig, h_orig = img.size
            scale_orig = w_orig / w_viewport

            # Hạ độ phân giải ảnh chụp xuống kích thước CSS chuẩn để tăng tốc Ollama 4 lần
            img = img.resize((w_viewport, h_viewport), Image.Resampling.LANCZOS)
            w, h = img.size
            scale = 1.0  # Đã chuyển đổi sang tỷ lệ CSS chuẩn

            # Áp dụng bộ lọc làm sắc nét và tăng tương phản trên ảnh đã thu nhỏ
            try:
                img = img.filter(ImageFilter.SHARPEN)
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.25)
            except Exception as enh_err:
                logger.debug("Image enhancement failed: %s", enh_err)

            rgb_img = img.convert("RGB")
            output_bytes = io.BytesIO()
            rgb_img.save(output_bytes, format="JPEG", quality=80)
            logger.info(
                "  [Downscale Viewport] Screenshot downscaled from %dx%d to %dx%d px (High-DPI original %.1fx -> target %.1fx, quality=80%%)",
                w_orig, h_orig, w, h, scale_orig, scale
            )

            # Tính toán tọa độ 4 góc của ảnh qua pixel_to_gps chính xác Web Mercator
            center_lat = lat
            center_lng = lng
            tl_lat, tl_lng = pixel_to_gps(
                center_lat=center_lat,
                center_lon=center_lng,
                zoom=SCREENSHOT_ZOOM,
                width=w,
                height=h,
                pixel_x=0,
                pixel_y=0
            )
            tr_lat, tr_lng = pixel_to_gps(
                center_lat=center_lat,
                center_lon=center_lng,
                zoom=SCREENSHOT_ZOOM,
                width=w,
                height=h,
                pixel_x=w,
                pixel_y=0
            )
            bl_lat, bl_lng = pixel_to_gps(
                center_lat=center_lat,
                center_lon=center_lng,
                zoom=SCREENSHOT_ZOOM,
                width=w,
                height=h,
                pixel_x=0,
                pixel_y=h
            )
            br_lat, br_lng = pixel_to_gps(
                center_lat=center_lat,
                center_lon=center_lng,
                zoom=SCREENSHOT_ZOOM,
                width=w,
                height=h,
                pixel_x=w,
                pixel_y=h
            )

            logger.info(
                "  [Tile Corners Reference]:\n"
                "    Top-Left:     (%.6f, %.6f)\n"
                "    Top-Right:    (%.6f, %.6f)\n"
                "    Bottom-Left:  (%.6f, %.6f)\n"
                "    Bottom-Right: (%.6f, %.6f)",
                tl_lat, tl_lng, tr_lat, tr_lng, bl_lat, bl_lng, br_lat, br_lng
            )

            img_metadata = {
                "width": w,
                "height": h,
                "center_x": w / 2.0,
                "center_y": h / 2.0,
                "scale": scale,
                "crop_x1": 0,
                "crop_y1": 0,
                "top_left_lat": tl_lat,
                "top_left_lng": tl_lng,
                "top_right_lat": tr_lat,
                "top_right_lng": tr_lng,
                "bottom_left_lat": bl_lat,
                "bottom_left_lng": bl_lng,
                "bottom_right_lat": br_lat,
                "bottom_right_lng": br_lng,
            }
            return screenshot_bytes, output_bytes.getvalue(), img_metadata
        except Exception as crop_err:
            logger.warning("Could not compress screenshot: %s", crop_err)

        # Fallback: trả ảnh PNG gốc
        try:
            from PIL import Image
            import io as _io
            _img = Image.open(_io.BytesIO(screenshot_bytes))
            w, h = _img.size
        except Exception:
            w, h = w_viewport * 2, h_viewport * 2

        center_lat = lat
        center_lng = lng
        tl_lat, tl_lng = pixel_to_gps(
            center_lat=center_lat,
            center_lon=center_lng,
            zoom=SCREENSHOT_ZOOM,
            width=w,
            height=h,
            pixel_x=0,
            pixel_y=0
        )
        tr_lat, tr_lng = pixel_to_gps(
            center_lat=center_lat,
            center_lon=center_lng,
            zoom=SCREENSHOT_ZOOM,
            width=w,
            height=h,
            pixel_x=w,
            pixel_y=0
        )
        bl_lat, bl_lng = pixel_to_gps(
            center_lat=center_lat,
            center_lon=center_lng,
            zoom=SCREENSHOT_ZOOM,
            width=w,
            height=h,
            pixel_x=0,
            pixel_y=h
        )
        br_lat, br_lng = pixel_to_gps(
            center_lat=center_lat,
            center_lon=center_lng,
            zoom=SCREENSHOT_ZOOM,
            width=w,
            height=h,
            pixel_x=w,
            pixel_y=h
        )

        img_metadata = {
            "width": w,
            "height": h,
            "center_x": w / 2.0,
            "center_y": h / 2.0,
            "scale": 2.0,
            "crop_x1": 0,
            "crop_y1": 0,
            "top_left_lat": tl_lat,
            "top_left_lng": tl_lng,
            "top_right_lat": tr_lat,
            "top_right_lng": tr_lng,
            "bottom_left_lat": bl_lat,
            "bottom_left_lng": bl_lng,
            "bottom_right_lat": br_lat,
            "bottom_right_lng": br_lng,
        }
        return screenshot_bytes, screenshot_bytes, img_metadata

    async def _save_screenshot(self, data: bytes, tx: int, ty: int) -> None:
        """Lưu screenshot ra disk (chỉ dùng khi debug)."""
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        path = os.path.join(SCREENSHOT_DIR, f"tile_{tx}_{ty}.png")
        try:
            with open(path, "wb") as f:
                f.write(data)
        except Exception as exc:
            logger.debug("Could not save screenshot: %s", exc)
