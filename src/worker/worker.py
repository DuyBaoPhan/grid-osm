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

from .browser import BrowserLifecycleMixin
from .runner import WorkerRunMixin
from .processing import TileProcessingMixin
from .dom import DomExtractionMixin
from .capture import CaptureMixin


class Worker(
    BrowserLifecycleMixin,
    WorkerRunMixin,
    TileProcessingMixin,
    DomExtractionMixin,
    CaptureMixin,
):
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
