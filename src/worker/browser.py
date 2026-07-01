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

class BrowserLifecycleMixin:
    async def _start_browser(self) -> None:
        """Khởi động Chromium instance nếu chưa có."""
        assert self._playwright is not None, "Playwright is not initialized"
        if self._browser is None:
            w_win = SCREENSHOT_W + 2 * SCREENSHOT_OVERLAP_PX
            h_win = SCREENSHOT_H + 2 * SCREENSHOT_OVERLAP_PX + 50
            self._browser = await self._playwright.chromium.launch(
                headless=HEADLESS,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-extensions",
                    "--disable-background-networking",
                    "--disable-default-apps",
                    f"--window-size={w_win},{h_win}",
                ],
            )
            self._tile_count = 0
            logger.info("[Worker %d] Browser started.", self.id)


    async def _start_page(self) -> None:
        """Tạo context mới và page mới sạch sẽ cho ô quét hiện tại."""
        await self._close_page()
        await self._start_browser()

        self._context = await self._browser.new_context(
            viewport={"width": SCREENSHOT_W + 2 * SCREENSHOT_OVERLAP_PX, "height": SCREENSHOT_H + 2 * SCREENSHOT_OVERLAP_PX},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            bypass_csp=True,
        )
        self._page = await self._context.new_page()

        # Inject CSS to hide Google Maps UI elements before they render
        await self._page.add_init_script("""
            const style = document.createElement('style');
            style.textContent = `
                /* Ẩn Google Maps UI: search bar, buttons, controls */
                .dismissal, .searchbox, .searchboxinput, .widget-settings-button,
                .app-viewcard-strip, .scene-footer, .watermark, .gm-style-cc,
                button[jsaction], .widget-expand-button,
                .app-vertical-scrollable-container, .omnibox-container,
                .app-viewcard-strip-container,
                .ml-promotion-container, .cards-module, #searchbox, #omnibox,
                #appbar, .scene-footer-container, .widget-settings, .widget-zoom,
                .gm-bundled-control, .gm-svpc, .gm-fullscreen-control,
                #searchboxinput, .fbar, [id="sb_cb"],
                .gb_Md, .gb_od, #gb, .gbh, .gba,
                [role="banner"], [role="dialog"], [role="alertdialog"],
                [role="search"], [role="navigation"] {
                    display: none !important;
                    visibility: hidden !important;
                    height: 0 !important;
                    max-height: 0 !important;
                    overflow: hidden !important;
                    pointer-events: none !important;
                }
                body, html {
                    overflow: hidden !important;
                    margin: 0 !important;
                    padding: 0 !important;
                }
            `;
            document.documentElement.appendChild(style);
        """)


    async def _close_page(self) -> None:
        """Đóng context và page hiện tại."""
        for obj in [self._page, self._context]:
            if obj is not None:
                try:
                    await obj.close()
                except Exception:
                    pass
        self._page = self._context = None


    async def _close_browser(self) -> None:
        """Đóng toàn bộ browser và dọn dẹp tài nguyên."""
        await self._close_page()
        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception:
                pass
        self._browser = None
