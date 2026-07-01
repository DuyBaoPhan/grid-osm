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

class TileProcessingMixin:
    async def _process_tile(self, tile: TileCoord) -> None:
        """
        Xử lý 1 tile:
          1. Khởi động browser
          2. Điều hướng đến Google Maps URL
          3. Chụp screenshot và trích xuất TOÀN BỘ tọa độ DOM địa điểm cùng một lúc
          4. Đóng browser ngay lập tức để tiết kiệm RAM, CPU và dọn dẹp màn hình!
          5. Gọi vision → danh sách POI (chạy ngầm 30-60s không cần browser)
          6. Báo kết quả cho coordinator
        Retry MAX_RETRIES lần nếu lỗi.
        """
        tx, ty = tile
        # Tile (0,0) dùng chính xác CENTER_LAT/CENTER_LNG để Bưu điện Trung tâm Sài Gòn nằm giữa màn hình
        if tx == 0 and ty == 0:
            lat, lng = CENTER_LAT, CENTER_LNG
        else:
            lat, lng = tile_center(tx, ty, ZOOM_LEVEL)
        strict_bbox = tile_viewport_bbox(tx, ty, ZOOM_LEVEL)
        url = _GMAP_URL.format(zoom=SCREENSHOT_ZOOM, lat=round(lat, 6), lng=round(lng, 6))
        logger.info("  [Navigate] URL: %s (zoom=%d)", url, SCREENSHOT_ZOOM)

        # Đảm bảo khởi động trình duyệt và page mới cho ô quét này
        await self._start_page()

        browser_coords = {}
        img_metadata = {}
        for attempt in range(1, MAX_RETRIES + 2):
            try:
                raw_screenshot, compressed_screenshot, img_metadata = await self._capture_screenshot(
                    url, strict_bbox, self.coord._boundary
                )
                # Trích xuất toàn bộ nhãn và tọa độ hiển thị trong DOM hiện tại
                browser_coords = await self._extract_all_visible_poi_coords_from_browser()
                # Báo cáo ngay cho coordinator rằng đã chụp ảnh xong để vẽ ô màu xanh neon blue lên bản đồ!
                await self.coord.report_captured(tile, strict_bbox)
                break
            except Exception as exc:
                if attempt <= MAX_RETRIES:
                    logger.warning(
                        "[Worker %d] Screenshot attempt %d/%d failed for tile (%d,%d): %s",
                        self.id, attempt, MAX_RETRIES, tx, ty, exc,
                    )
                    await asyncio.sleep(attempt * 2.0)
                    try:
                        await self._close_browser()
                        await self._start_page()
                    except Exception:
                        pass
                else:
                    logger.error(
                        "[Worker %d] Tile (%d,%d) skipped after %d attempts.",
                        self.id, tx, ty, MAX_RETRIES + 1,
                    )
                    await self._close_page()
                    # Re-queue để thử lại trong phiên sau
                    await self.coord._queue.put(tile)
                    return

        # Detect-first pipeline: giữ nguyên nền bản đồ gốc để detect chính xác nhất.
        enhanced_screenshot = enhance_for_detection(raw_screenshot)

        # Nhận diện POI trên ảnh FULL có overlap để không cắt mất nhãn nằm sát mép vùng quét.
        # Tọa độ OCR lúc này đã là tọa độ ảnh full, nên crop offset phải = 0.
        vision_metadata = dict(img_metadata)
        vision_metadata["core_x1"] = float(img_metadata.get("crop_x1", 0.0))
        vision_metadata["core_y1"] = float(img_metadata.get("crop_y1", 0.0))
        vision_metadata["core_x2"] = float(img_metadata.get("crop_x1", 0.0)) + float(SCREENSHOT_W)
        vision_metadata["core_y2"] = float(img_metadata.get("crop_y1", 0.0)) + float(SCREENSHOT_H)
        vision_metadata["crop_x1"] = 0.0
        vision_metadata["crop_y1"] = 0.0
        poi_names, outside_district = await extract_pois_from_screenshot(
            raw_screenshot, tx=tx, ty=ty, img_metadata=vision_metadata
        )
        core_l = float(vision_metadata.get("core_x1", img_metadata.get("crop_x1", 0.0)))
        core_t = float(vision_metadata.get("core_y1", img_metadata.get("crop_y1", 0.0)))
        core_r = float(vision_metadata.get("core_x2", core_l + SCREENSHOT_W))
        core_b = float(vision_metadata.get("core_y2", core_t + SCREENSHOT_H))
        before_filter = len(poi_names)
        poi_names = [
            p for p in poi_names
            if p.get("x") is not None and p.get("y") is not None
            and core_l <= float(p.get("x")) <= core_r
            and core_t <= float(p.get("y")) <= core_b
        ]
        if len(poi_names) != before_filter:
            logger.info(
                "  [OverlapFilter] Kept %d/%d POIs inside core tile bounds",
                len(poi_names), before_filter,
            )
        logger.info("  [2/2] YOLOv8 + VietOCR Detection Done.")

        # Dùng overlap screenshot để giảm nhãn bị cắt mép; không chụp rescue phụ để tránh quét lại.
        # poi_names = await self._rescue_edge_cut_pois(
        #     poi_names,
        #     center_lat=lat,
        #     center_lng=lng,
        #     tx=tx,
        #     ty=ty,
        # )

        # Lưu screenshot: vẽ khung đỏ trực tiếp lên ảnh để giám sát
        if SAVE_SCREENSHOTS:
            debug_pois = []
            debug_dx = float(img_metadata.get("crop_x1", 0.0))
            debug_dy = float(img_metadata.get("crop_y1", 0.0))
            for p in poi_names:
                dp = dict(p)
                if dp.get("x") is not None:
                    dp["x"] = float(dp["x"]) - debug_dx
                if dp.get("y") is not None:
                    dp["y"] = float(dp["y"]) - debug_dy
                bbox = dp.get("bbox")
                if bbox and len(bbox) >= 4:
                    dp["bbox"] = [
                        float(bbox[0]) - debug_dx,
                        float(bbox[1]) - debug_dy,
                        float(bbox[2]),
                        float(bbox[3]),
                    ]
                debug_pois.append(dp)
            final_img = draw_detections(compressed_screenshot, debug_pois)
            await self._save_screenshot(final_img, tx, ty)

        # Đóng page và context để giải phóng tài nguyên sau khi quét xong ô này
        await self._close_page()

        # Lọc và khớp tọa độ địa điểm
        lat_min, lng_min, lat_max, lng_max = strict_bbox

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

                # 1. Pixel exact từ vision.py là nguồn chân lý:
                #    has_icon=True  -> tâm bbox icon.
                #    has_icon=False -> tâm bbox text/label.
                # DOM chỉ fallback nếu OCR không trả x/y.
                has_ocr_xy = item.get("x") is not None and item.get("y") is not None
                if has_ocr_xy:
                    x_val = float(item.get("x"))
                    y_val = float(item.get("y"))
                    # OCR đang chạy trên ảnh full có overlap, nên x/y đã là tọa độ full viewport.
                    # Không cộng crop_x1/crop_y1; chỉ debug overlay mới trừ offset khi vẽ lên tile crop.
                    x_phys = x_val
                    y_phys = y_val
                    poi_lat = None
                    poi_lng = None
                elif dom_match:
                    # Fallback hiếm: nếu OCR thiếu pixel, dùng DOM.
                    cx = dom_match.get("cx", center_x / scale)
                    cy = dom_match.get("cy", center_y / scale)
                    x_phys = cx * scale
                    y_phys = cy * scale
                    poi_lat = dom_match["lat"]
                    poi_lng = dom_match["lng"]
                else:
                    x_phys = center_x
                    y_phys = center_y
                    poi_lat = None
                    poi_lng = None

                # 2. Tính khoảng cách pixel vật lý (distance_pixels) từ tâm
                distance_pixels = math.sqrt((x_phys - center_x)**2 + (y_phys - center_y)**2)

                # 3. Tính bearing từ tâm theo pixel
                dx = x_phys - center_x
                dy = center_y - y_phys  # Trục Oy hướng lên (Bắc) là dương, pixel y đi xuống
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

                if poi_lat is None or poi_lng is None:
                    # Quy đổi kích thước ảnh và vị trí pixel từ vật lý sang CSS pixels trước khi tính toán
                    width_css = img_w / scale
                    height_css = img_h / scale
                    pixel_x_css = x_phys / scale
                    pixel_y_css = y_phys / scale

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
                        "  [GeoPixelExact] Resolved '%s' from OCR pixel center → x=%d y=%d (phys_x=%d phys_y=%d) dist_px=%.1f dist_m=%.1fm bearing=%.1fdeg → (%.6f, %.6f)",
                        name, int(item.get("x", 0)), int(item.get("y", 0)), int(x_phys), int(y_phys), distance_pixels, distance_meters, bearing_deg, poi_lat, poi_lng
                    )
                else:
                    logger.info(
                        "  [LeafletDOM-Fallback] Resolved '%s' via DOM fallback → x=%d y=%d dist_px=%.1f dist_m=%.1fm bearing=%.1fdeg → (%.6f, %.6f)",
                        name, int(x_val), int(y_val), distance_pixels, distance_meters, bearing_deg, poi_lat, poi_lng
                    )

            except Exception as geo_err:
                logger.warning("  [GeoPixel] Lỗi khi tính toán tọa độ và khoảng cách: %s. Trở về center GPS mặc định.", geo_err)
                poi_lat = lat
                poi_lng = lng
                distance_pixels = 0.0
                distance_meters = 0.0
                bearing_deg = 0.0

            canonical_match = resolve_canonical_name(
                name,
                name,
                browser_coords,
                poi_lat=poi_lat,
                poi_lng=poi_lng,
            )
            suggested_name = canonical_match.selected if canonical_match.action == "use_canonical" else ""
            if suggested_name and suggested_name != name:
                logger.info(
                    "  [CanonicalSuggestion] OCR='%s' suggested='%s' source=%s score=%.3f",
                    name,
                    suggested_name,
                    canonical_match.source,
                    canonical_match.score,
                )
            if canonical_match.needs_review:
                logger.info(
                    "  [NeedsReview] OCR='%s' best_score=%.3f reason=%s",
                    name,
                    canonical_match.score,
                    canonical_match.reason,
                )

            # Giữ text OCR trong crop làm nguồn chân lý. Canonical/nearby chỉ là gợi ý, không overwrite `name`.
            pois.append({
                "name":             name,
                "ocr_name":         name,
                "suggested_canonical_name": suggested_name,
                "name_source":      "ocr",
                "name_match_score": round(canonical_match.score, 3),
                "needs_review":     canonical_match.needs_review,
                "review_reason":    canonical_match.reason,
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


    async def _rescue_edge_cut_pois(
        self,
        poi_names: List[dict],
        *,
        center_lat: float,
        center_lng: float,
        tx: int,
        ty: int,
    ) -> List[dict]:
        """Chụp lại view đã dịch tâm cho POI bị cắt mép và dùng OCR tốt hơn nếu có."""
        edge_items = [p for p in poi_names if p.get("edge_cut")]
        if not edge_items:
            return poi_names

        # Giới hạn mỗi tile để tránh rescue làm chậm toàn bộ scan khi có nhiều label sát mép.
        max_rescues = 1
        for item in edge_items[:max_rescues]:
            old_name = item.get("name", "").strip()
            edge_sides = item.get("edge_sides") or []
            try:
                rescue_lat, rescue_lng = self._compute_rescue_center(
                    center_lat, center_lng, edge_sides
                )
                rescue_bbox = self._bbox_for_center(rescue_lat, rescue_lng)
                rescue_url = _GMAP_URL.format(
                    zoom=SCREENSHOT_ZOOM,
                    lat=round(rescue_lat, 6),
                    lng=round(rescue_lng, 6),
                )
                logger.info(
                    "  [EdgeRescue] Re-capturing clipped POI '%s' sides=%s at (%.6f, %.6f)",
                    old_name, ",".join(edge_sides), rescue_lat, rescue_lng,
                )
                _raw, rescue_img, rescue_meta = await self._capture_screenshot(
                    rescue_url, rescue_bbox, self.coord._boundary
                )
                rescue_pois, _ = await extract_pois_from_screenshot(
                    rescue_img, tx=tx, ty=ty, img_metadata=rescue_meta
                )
                best = self._select_rescue_candidate(old_name, rescue_pois)
                if best:
                    new_name = best.get("name", "").strip()
                    item["original_edge_name"] = old_name
                    item["name"] = new_name
                    item["rescued_from_edge"] = True
                    item["rescue_edge_sides"] = edge_sides
                    item["rescue_confidence"] = best.get("confidence")
                    logger.info("  [EdgeRescue] '%s' -> '%s'", old_name, new_name)
            except Exception as exc:
                logger.warning("  [EdgeRescue] Failed for '%s': %s", old_name, exc)
        return poi_names


    def _compute_rescue_center(self, lat: float, lng: float, edge_sides: List[str]) -> Tuple[float, float]:
        """Dịch tâm map về phía mép bị cắt để label quay vào giữa ảnh hơn."""
        width_css = SCREENSHOT_W + 2 * SCREENSHOT_OVERLAP_PX
        height_css = SCREENSHOT_H + 2 * SCREENSHOT_OVERLAP_PX
        px = width_css / 2.0
        py = height_css / 2.0
        shift_x = width_css * 0.28
        shift_y = height_css * 0.28
        if "right" in edge_sides:
            px += shift_x
        if "left" in edge_sides:
            px -= shift_x
        if "top" in edge_sides:
            py -= shift_y
        if "bottom" in edge_sides:
            py += shift_y
        return pixel_to_gps(lat, lng, SCREENSHOT_ZOOM, width_css, height_css, px, py)


    def _bbox_for_center(self, lat: float, lng: float) -> Tuple[float, float, float, float]:
        """Tạo bbox metadata quanh center rescue bằng kích thước viewport hiện tại."""
        width_css = SCREENSHOT_W + 2 * SCREENSHOT_OVERLAP_PX
        height_css = SCREENSHOT_H + 2 * SCREENSHOT_OVERLAP_PX
        tl_lat, tl_lng = pixel_to_gps(lat, lng, SCREENSHOT_ZOOM, width_css, height_css, 0, 0)
        br_lat, br_lng = pixel_to_gps(lat, lng, SCREENSHOT_ZOOM, width_css, height_css, width_css, height_css)
        return min(br_lat, tl_lat), min(tl_lng, br_lng), max(br_lat, tl_lat), max(tl_lng, br_lng)


    def _select_rescue_candidate(self, old_name: str, candidates: List[dict]) -> Optional[dict]:
        """Chọn OCR rescue tốt hơn: dài hơn, không Unknown, ưu tiên cùng prefix/suffix bỏ dấu."""
        old_clean = self._name_key(old_name)
        if not old_clean:
            return None
        best = None
        best_score = 0.0
        for cand in candidates:
            name = cand.get("name", "").strip()
            key = self._name_key(name)
            if not key or key.startswith("unknown"):
                continue
            length_gain = len(key) - len(old_clean)
            contains = old_clean in key or key in old_clean
            prefix = key[:8] == old_clean[:8] if len(old_clean) >= 8 and len(key) >= 8 else False
            score = length_gain + (20 if contains else 0) + (12 if prefix else 0)
            if "..." in old_name or "…" in old_name:
                score += 10
            if len(key) <= len(old_clean) + 2 and not contains:
                continue
            if score > best_score:
                best = cand
                best_score = score
        return best


    def _name_key(self, text: str) -> str:
        try:
            from src.vision import _strip_vietnamese_accents
            text = _strip_vietnamese_accents(text or "")
        except Exception:
            text = (text or "").lower()
        return re.sub(r"[^a-z0-9]+", "", text.lower())
