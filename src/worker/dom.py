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

class DomExtractionMixin:
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
