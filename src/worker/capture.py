# =============================================================
# Extracted from original worker.py. Method bodies kept unchanged.
# =============================================================

import asyncio
import logging
import os
import re
from typing import List, Tuple, Optional

from playwright.async_api import Browser, BrowserContext, Page, Playwright

from config import (
    BROWSER_RESTART_EVERY,
    CENTER_LAT,
    CENTER_LNG,
    DELAY_BETWEEN_REQ,
    MAX_RETRIES,
    PAGE_LOAD_TIMEOUT,
    PAGE_SETTLE_MS,
    SAVE_SCREENSHOTS,
    SCREENSHOT_DIR,
    SCREENSHOT_H,
    SCREENSHOT_W,
    SCREENSHOT_OVERLAP_PX,
    SCREENSHOT_ZOOM,
    ZOOM_LEVEL,
    HEADLESS,
    SAVE_POI_CROPS,
    POI_CROPS_DIR,
)
from grid import tile_center, tile_bbox, tile_viewport_bbox, pixel_to_gps
from src.vision import extract_pois_from_screenshot, draw_detections, enhance_for_detection
from src.canonical_matcher import resolve_canonical_name

logger = logging.getLogger(__name__)

TileCoord = Tuple[int, int]
_GMAP_URL = "https://www.google.com/maps/@{lat},{lng},{zoom}z"

class CaptureMixin:
    async def _capture_screenshot(
        self,
        url: str,
        bbox: Tuple[float, float, float, float],
        boundary_geometry: Optional[dict] = None
    ) -> Tuple[bytes, bytes, dict]:
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

        # Nhấn Escape để đóng popup/card panel
        try:
            await self._page.wait_for_timeout(2000)
            await self._page.keyboard.press('Escape')
            await self._page.wait_for_timeout(500)
            await self._page.keyboard.press('Escape')
            await self._page.wait_for_timeout(300)
        except Exception:
            pass

        # 2. JavaScript: ẩn UI overlay và chờ Google Maps load
        crop_box = None
        try:
            crop_box = await self._page.evaluate(f"""async () => {{

                // ── 1. Click nút đóng (X) của bất kỳ panel/card nào đang mở ──────────
                const closeBtns = document.querySelectorAll(
                    '[aria-label="Close"], [aria-label="Đóng"], [aria-label="close"], '
                    + 'button[jsaction*="dismiss"], button[jsaction*="close"], '
                    + '[data-dismiss], [jsaction*="panel.close"]'
                );
                closeBtns.forEach(btn => {{ try {{ btn.click(); }} catch(e) {{}} }});
                await new Promise(resolve => setTimeout(resolve, 800));

                // ── 2. Hàm ẩn element theo vùng vị trí ──────────────────────────────────
                const HEADER_H = 160;   // search bar + category tabs (Restaurants, Hotels...)
                const PANEL_W  = 450;   // left panel card (This area, POI detail, etc.)
                const SKIP_TAGS = new Set(['CANVAS','SCRIPT','STYLE','HTML','BODY','HEAD','IMG']);

                function hideOverlayElements() {{
                    document.querySelectorAll('body *').forEach(el => {{
                        if (SKIP_TAGS.has(el.tagName)) return;
                        const rect = el.getBoundingClientRect();
                        if (rect.width === 0 || rect.height === 0) return;
                        const inHeader    = rect.top >= 0 && rect.bottom <= HEADER_H && rect.width > 60;
                        const inLeftPanel = rect.left >= 0 && rect.right <= PANEL_W  && rect.height > 30;
                        if (inHeader || inLeftPanel) {{
                            el.style.setProperty('display',        'none',   'important');
                            el.style.setProperty('visibility',     'hidden', 'important');
                            el.style.setProperty('pointer-events', 'none',   'important');
                            el.style.setProperty('height',         '0',      'important');
                            el.style.setProperty('overflow',       'hidden', 'important');
                        }}
                    }});
                    // Ẩn thêm bằng selector ngữ nghĩa
                    [
                        '[role="search"]', '[role="navigation"]', '[role="banner"]',
                        '[role="dialog"]', '[role="alertdialog"]',
                        'header', 'nav', '.searchbox', '#searchboxinput',
                        '.app-viewcard-strip', '.scene-footer',
                        '[aria-label="Search Google Maps"]', 'form'
                    ].forEach(sel => {{
                        document.querySelectorAll(sel).forEach(el => {{
                            el.style.setProperty('display', 'none', 'important');
                        }});
                    }});
                }}

                // ── 3. Chạy ngay lập tức ─────────────────────────────────────────────────
                hideOverlayElements();


                // ── 4. MutationObserver: ẩn liên tục khi Google Maps tạo element mới ────
                // (Google Maps SPA tái tạo category tabs sau mỗi lần re-render)
                const observer = new MutationObserver(() => hideOverlayElements());
                observer.observe(document.body, {{
                    childList: true,
                    subtree: true,
                    attributes: false
                }});

                // Backup interval mỗi 500ms phòng khi MutationObserver bỏ sót
                const hideInterval = setInterval(hideOverlayElements, 500);

                // ── 5. Dispatch resize để Google Maps fill viewport ───────────────────────
                window.dispatchEvent(new Event('resize'));
                await new Promise(resolve => setTimeout(resolve, 1000));
                window.dispatchEvent(new Event('resize'));

                // ── 6. Chờ lâu hơn cho map load (tăng từ 10s lên 20s vì ảnh đang trắng) ─────────
                await new Promise(resolve => setTimeout(resolve, 20000));

                // ── 7. Dừng observer + interval, chạy hide lần cuối ─────────────────────
                clearInterval(hideInterval);
                observer.disconnect();
                hideOverlayElements();

                console.log('[CENTER]', {{ lat: {lat}, lng: {lng} }});

                // ── 8. Không vẽ scan box vào DOM trước khi chụp screenshot ───────────────
                // Trước đây code tạo `active-worker-scan-box` với border cyan tại đây.
                // Vì screenshot chụp toàn viewport sau bước này, border đó trở thành pixel thật
                // và lọt vào crop/OCR như các đường xanh ngang qua nhãn POI.
                // Nếu cần debug vùng tile, hãy vẽ overlay sau khi chụp trên ảnh debug riêng,
                // không vẽ trực tiếp lên Google Maps DOM trước screenshot.
                const oldScanBox = document.getElementById('active-worker-scan-box');
                if (oldScanBox) oldScanBox.remove();

                return null;  // Full viewport, không crop
            }}""")
        except Exception as exc:
            logger.debug("Could not remove OSM elements or draw scan box via JS: %s", exc)

        try:
            await self._page.wait_for_load_state("networkidle", timeout=4000)
        except Exception:
            pass
        await self._page.wait_for_timeout(PAGE_SETTLE_MS)

        # Get actual viewport size dynamically via Javascript since page.viewport_size is None when no_viewport=True
        w_viewport = SCREENSHOT_W + 2 * SCREENSHOT_OVERLAP_PX
        h_viewport = SCREENSHOT_H + 2 * SCREENSHOT_OVERLAP_PX
        try:
            v_size = await self._page.evaluate("() => ({ width: window.innerWidth, height: window.innerHeight })")
            if v_size and "width" in v_size and "height" in v_size:
                w_viewport = int(v_size["width"])
                h_viewport = int(v_size["height"])
        except Exception as eval_exc:
            logger.debug("Failed to evaluate viewport size via JS: %s", eval_exc)

        screenshot_bytes = await self._page.screenshot(type="png")

        # 3. Tính toán thông tin ảnh mà không co dãn (no downscale) để giữ nguyên độ phân giải và nét chữ
        try:
            from PIL import Image
            import io

            img = Image.open(io.BytesIO(screenshot_bytes))
            w_orig, h_orig = img.size
            scale = w_orig / w_viewport  # device scale factor (tỷ lệ điểm ảnh vật lý / CSS)
            w, h = w_orig, h_orig

            # Tính toán tọa độ 4 góc của ảnh qua pixel_to_gps chính xác Web Mercator
            # Lưu ý: pixel_to_gps cần các tham số width, height, pixel_x, pixel_y ở kích thước CSS
            center_lat = lat
            center_lng = lng
            tl_lat, tl_lng = pixel_to_gps(
                center_lat=center_lat,
                center_lon=center_lng,
                zoom=SCREENSHOT_ZOOM,
                width=w_viewport,
                height=h_viewport,
                pixel_x=0,
                pixel_y=0
            )
            tr_lat, tr_lng = pixel_to_gps(
                center_lat=center_lat,
                center_lon=center_lng,
                zoom=SCREENSHOT_ZOOM,
                width=w_viewport,
                height=h_viewport,
                pixel_x=w_viewport,
                pixel_y=0
            )
            bl_lat, bl_lng = pixel_to_gps(
                center_lat=center_lat,
                center_lon=center_lng,
                zoom=SCREENSHOT_ZOOM,
                width=w_viewport,
                height=h_viewport,
                pixel_x=0,
                pixel_y=h_viewport
            )
            br_lat, br_lng = pixel_to_gps(
                center_lat=center_lat,
                center_lon=center_lng,
                zoom=SCREENSHOT_ZOOM,
                width=w_viewport,
                height=h_viewport,
                pixel_x=w_viewport,
                pixel_y=h_viewport
            )

            logger.info(
                "  [Capture Screenshot] Keep original resolution %dx%d px (DPI scale = %.2fx)",
                w, h, scale
            )
            logger.info(
                "  [Tile Corners Reference]:\n"
                "    Top-Left:     (%.6f, %.6f)\n"
                "    Top-Right:    (%.6f, %.6f)\n"
                "    Bottom-Left:  (%.6f, %.6f)\n"
                "    Bottom-Right: (%.6f, %.6f)",
                tl_lat, tl_lng, tr_lat, tr_lng, bl_lat, bl_lng, br_lat, br_lng
            )

            # Crop ảnh lưu về đúng kích thước tile (SCREENSHOT_W × SCREENSHOT_H)
            # để ảnh chụp = đúng y chang vùng hiển thị trên trình duyệt
            target_w = int(SCREENSHOT_W * scale)
            target_h = int(SCREENSHOT_H * scale)
            cx_phys  = w_orig // 2
            cy_phys  = h_orig // 2
            crop_l = max(0, cx_phys - target_w // 2)
            crop_t = max(0, cy_phys - target_h // 2)
            crop_r = min(w_orig, crop_l + target_w)
            crop_b = min(h_orig, crop_t + target_h)

            img_metadata = {
                "width": w,
                "height": h,
                "center_x": w / 2.0,
                "center_y": h / 2.0,
                "scale": scale,
                "crop_x1": crop_l,
                "crop_y1": crop_t,
                "top_left_lat": tl_lat,
                "top_left_lng": tl_lng,
                "top_right_lat": tr_lat,
                "top_right_lng": tr_lng,
                "bottom_left_lat": bl_lat,
                "bottom_left_lng": bl_lng,
                "bottom_right_lat": br_lat,
                "bottom_right_lng": br_lng,
            }

            img_tile = img.crop((crop_l, crop_t, crop_r, crop_b))
            buf_tile = io.BytesIO()
            img_tile.save(buf_tile, format="PNG")
            compressed_screenshot = buf_tile.getvalue()
            logger.info(
                "  [Tile Crop] Full=%dx%d → Tile=%dx%d (crop=%d,%d,%d,%d)",
                w_orig, h_orig, crop_r - crop_l, crop_b - crop_t,
                crop_l, crop_t, crop_r, crop_b
            )

            return screenshot_bytes, compressed_screenshot, img_metadata
        except Exception as crop_err:
            logger.warning("Could not process screenshot metadata: %s", crop_err)

        # Fallback: trả ảnh PNG gốc
        try:
            from PIL import Image
            import io as _io
            _img = Image.open(_io.BytesIO(screenshot_bytes))
            w_orig, h_orig = _img.size
            scale = w_orig / w_viewport
        except Exception:
            w_orig, h_orig = w_viewport * 2, h_viewport * 2
            scale = 2.0

        center_lat = lat
        center_lng = lng
        tl_lat, tl_lng = pixel_to_gps(
            center_lat=center_lat,
            center_lon=center_lng,
            zoom=SCREENSHOT_ZOOM,
            width=w_viewport,
            height=h_viewport,
            pixel_x=0,
            pixel_y=0
        )
        tr_lat, tr_lng = pixel_to_gps(
            center_lat=center_lat,
            center_lon=center_lng,
            zoom=SCREENSHOT_ZOOM,
            width=w_viewport,
            height=h_viewport,
            pixel_x=w_viewport,
            pixel_y=0
        )
        bl_lat, bl_lng = pixel_to_gps(
            center_lat=center_lat,
            center_lon=center_lng,
            zoom=SCREENSHOT_ZOOM,
            width=w_viewport,
            height=h_viewport,
            pixel_x=0,
            pixel_y=h_viewport
        )
        br_lat, br_lng = pixel_to_gps(
            center_lat=center_lat,
            center_lon=center_lng,
            zoom=SCREENSHOT_ZOOM,
            width=w_viewport,
            height=h_viewport,
            pixel_x=w_viewport,
            pixel_y=h_viewport
        )

        img_metadata = {
            "width": w_orig,
            "height": h_orig,
            "center_x": w_orig / 2.0,
            "center_y": h_orig / 2.0,
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
