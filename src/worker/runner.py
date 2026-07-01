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

class WorkerRunMixin:
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
                    await self._close_browser()
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
