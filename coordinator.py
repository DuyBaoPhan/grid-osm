# =============================================================
# coordinator.py — Quản lý hàng đợi tile, checkpoint và kết quả
#
# Chức năng:
#   - Khởi tạo danh sách tile theo ranh giới OSM polygon của quận
#   - Quản lý queue xử lý tile theo thứ tự outward BFS từ trung tâm
#   - Kiểm tra geofence: chỉ giữ POI có tọa độ TRONG polygon quận
#   - Geo-override: ghi đè LLM hallucination "outside_district" nếu
#     tile THỰC SỰ nằm trong polygon → ngăn tạo lỗ hổng quét lớn
#   - Lưu checkpoint và results định kỳ ra disk
#   - Cập nhật map_viewer.html sau mỗi tile được xử lý
# =============================================================

import asyncio
import json
import logging
import os
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from config import (
    CENTER_LAT,
    CENTER_LNG,
    CHECKPOINT_FILE,
    EXPAND_EMPTY,
    NUM_WORKERS,
    RADIUS_KM,
    RESULTS_FILE,
    TARGET_DISTRICT,
    ZOOM_LEVEL,
)
from grid import (
    generate_all_tiles,
    is_point_in_boundary,
    lat_lng_to_tile,
    load_or_download_boundary,
    tile_center,
)

logger = logging.getLogger(__name__)

TileCoord = Tuple[int, int]


class Coordinator:
    """
    Điều phối toàn bộ quá trình quét tile.

    Trạng thái nội bộ:
        _all_tiles  : tập hợp tất cả tile hợp lệ trong quận
        _visited    : tập hợp tile đã xử lý xong
        _discarded  : tập hợp tile bị loại bỏ (ngoài ranh giới)
        _queued     : tập hợp tile đang ở trong queue (chờ xử lý)
        _queue      : asyncio.Queue chứa các tile cần xử lý
        _results    : list POI đã thu thập
        _boundary   : GeoJSON geometry của ranh giới quận (dùng để geofence)
        _lock       : asyncio.Lock để bảo vệ trạng thái chia sẻ giữa các worker
    """

    def __init__(self) -> None:
        self._all_tiles: List[TileCoord] = []
        self._all_tiles_set: Set[TileCoord] = set()
        self._visited: Set[TileCoord] = set()
        self._discarded: Set[TileCoord] = set()
        self._queued: Set[TileCoord] = set()
        self._queue: asyncio.Queue = asyncio.Queue()
        self._results: List[dict] = []
        self._boundary: Optional[dict] = None
        self._lock = asyncio.Lock()

    # ── Khởi tạo ────────────────────────────────────────────

    async def init(self) -> None:
        """
        Khởi tạo coordinator:
          1. Tải ranh giới hành chính quận từ Nominatim/cache
          2. Sinh danh sách tile hợp lệ
          3. Nạp checkpoint (nếu có) để tiếp tục từ lần chạy trước
          4. Đưa tile khởi điểm vào queue
        """
        # 1. Tải ranh giới quận
        logger.info("Loading district boundary for '%s'...", TARGET_DISTRICT)
        self._boundary = load_or_download_boundary(TARGET_DISTRICT)
        if self._boundary:
            logger.info("District boundary loaded successfully.")
        else:
            logger.warning(
                "Could not load district boundary for '%s'. "
                "Geofencing will be disabled — all tiles within radius will be scanned.",
                TARGET_DISTRICT,
            )

        # 2. Sinh danh sách tile
        self._all_tiles = generate_all_tiles(
            CENTER_LAT, CENTER_LNG, RADIUS_KM, ZOOM_LEVEL, TARGET_DISTRICT
        )
        self._all_tiles_set = set(self._all_tiles)
        logger.info("Total tiles to scan: %d", len(self._all_tiles))

        # 3. Nạp checkpoint
        visited_from_checkpoint, queued_from_checkpoint = self._load_checkpoint()
        self._results = self._load_results()

        # 4. Xác định tile ban đầu (tile chứa tâm quận)
        center_tile = lat_lng_to_tile(CENTER_LAT, CENTER_LNG, ZOOM_LEVEL)
        if center_tile not in self._all_tiles_set:
            # Nếu tâm không trong danh sách, lấy tile gần tâm nhất
            if self._all_tiles:
                center_tile = _sort_by_distance(set(self._all_tiles), CENTER_LAT, CENTER_LNG)[0]

        # 5. Khởi tạo queue từ checkpoint nếu có, hoặc bắt đầu từ tâm
        if visited_from_checkpoint or queued_from_checkpoint:
            self._visited = visited_from_checkpoint & self._all_tiles_set
            self._discarded = self._load_discarded()

            # Đưa lại các tile đã queued từ checkpoint vào queue
            seed_tiles = queued_from_checkpoint - self._visited - self._discarded
            # Nếu queue từ checkpoint trống, tìm tile biên chưa xử lý
            if not seed_tiles:
                seed_tiles = self._find_frontier_tiles()
            if not seed_tiles:
                remaining = self._all_tiles_set - self._visited - self._discarded
                seed_tiles = set(_sort_by_distance(remaining, CENTER_LAT, CENTER_LNG)[:NUM_WORKERS])

            for tile in _sort_by_distance(seed_tiles, CENTER_LAT, CENTER_LNG):
                if tile not in self._visited and tile not in self._discarded:
                    await self._queue.put(tile)
                    self._queued.add(tile)
            logger.info(
                "Resumed from checkpoint: %d done, %d discarded, %d in queue",
                len(self._visited), len(self._discarded), self._queue.qsize(),
            )
        else:
            # Bắt đầu mới: seed đủ NUM_WORKERS tile, bắt đầu từ trung tâm toả ra
            # → mỗi worker đều có tile ngay từ đầu, không bị shutdown vì đói tile
            seed_candidates = _sort_by_distance(self._all_tiles_set, CENTER_LAT, CENTER_LNG)
            # Đảm bảo tile trung tâm luôn là tile đầu tiên
            seeds: list[TileCoord] = []
            if center_tile in self._all_tiles_set:
                seeds.append(center_tile)
            for t in seed_candidates:
                if t != center_tile and len(seeds) < NUM_WORKERS:
                    seeds.append(t)

            for tile in seeds:
                await self._queue.put(tile)
                self._queued.add(tile)
            logger.info(
                "Starting fresh scan from center tile %s with %d seed tiles. Total: %d tiles",
                center_tile, len(seeds), len(self._all_tiles),
            )

    # ── Queue management ─────────────────────────────────────

    async def get_next_tile(self) -> Optional[TileCoord]:
        """
        Lấy tile tiếp theo từ queue.
        Trả về None nếu queue rỗng và không còn tile nào để xử lý.

        Quy tắc fill-frontier:
          - Chỉ thêm tile vào queue khi ĐÃ có tile được xử lý (_visited không rỗng)
            → tránh race condition lúc khởi động: worker 2 gọi trước khi worker 1
            hoàn thành tile trung tâm và expand neighbors, dẫn đến tile ngẫu nhiên
            được chọn thay vì hàng xóm của tâm.
          - Nếu _visited còn rỗng (lần đầu, chỉ có seed trong queue), trả về None
            để worker ngủ 2s và thử lại (worker.py đã xử lý việc này).
        """
        async with self._lock:
            if self._queue.empty() and self._visited:
                # Đã có tile được xử lý → an toàn để tìm frontier thực sự
                remaining = self._all_tiles_set - self._visited - self._discarded - self._queued
                if remaining:
                    frontier = self._find_frontier_tiles()
                    if frontier:
                        # Lấy tile frontier gần trung tâm nhất
                        for tile in _sort_by_distance(frontier, CENTER_LAT, CENTER_LNG):
                            await self._queue.put(tile)
                            self._queued.add(tile)
                            break
                    # Nếu không có frontier (island tile) → fallback tile gần tâm nhất
                    elif remaining:
                        tile_fb = _sort_by_distance(remaining, CENTER_LAT, CENTER_LNG)[0]
                        await self._queue.put(tile_fb)
                        self._queued.add(tile_fb)

        try:
            tile = self._queue.get_nowait()
            return tile
        except asyncio.QueueEmpty:
            return None

    async def report_result(
        self,
        tile: TileCoord,
        pois: List[dict],
        neighbors: List[TileCoord],
        outside_district: bool = False,
    ) -> None:
        """
        Nhận báo cáo kết quả từ worker sau khi xử lý 1 tile.

        Args:
            tile            : tile đã xử lý
            pois            : danh sách POI đã tìm thấy (đã được lọc bbox bởi worker)
            neighbors       : danh sách 8 tile hàng xóm
            outside_district: cờ LLM báo tile nằm ngoài quận
        """
        async with self._lock:
            tx, ty = tile

            # ── Geo-override: kiểm tra chương trình xem tile có THỰC SỰ
            #    nằm trong polygon không, để ghi đè LLM hallucination ──
            geo_inside = True  # Mặc định: coi là trong quận
            if self._boundary:
                clat, clng = tile_center(tx, ty, ZOOM_LEVEL)
                geo_inside = is_point_in_boundary(clat, clng, self._boundary)

            if outside_district and geo_inside:
                logger.info(
                    "  [GeoOverride] LLM said outside but tile (%d,%d) IS inside polygon → overriding!",
                    tx, ty,
                )
                outside_district = False

            # ── Xử lý tile bị loại (thực sự ngoài quận) ──
            if outside_district and not geo_inside:
                self._discarded.add(tile)
                self._visited.add(tile)  # Đánh dấu đã xử lý
                logger.info(
                    "  Tile (%d,%d) discarded — confirmed outside district boundary.", tx, ty
                )
                await self._save_checkpoint()
                self._update_map()
                return

            # ── Lọc POI qua geofence polygon của quận ──
            filtered_pois = []
            for poi in pois:
                poi_lat = poi.get("approx_lat")
                poi_lng = poi.get("approx_lng")
                if poi_lat is None or poi_lng is None:
                    filtered_pois.append(poi)
                    continue
                if self._boundary:
                    if is_point_in_boundary(poi_lat, poi_lng, self._boundary):
                        filtered_pois.append(poi)
                    else:
                        logger.info(
                            "  [Geofence] Bỏ POI ngoài ranh giới polygon: %s (%.6f, %.6f)",
                            poi.get("name", "?"), poi_lat, poi_lng,
                        )
                else:
                    # Không có boundary → giữ tất cả (đã được lọc bbox bởi worker)
                    filtered_pois.append(poi)

            # ── Cập nhật trạng thái ──
            self._visited.add(tile)
            self._results.extend(filtered_pois)

            if filtered_pois:
                logger.info(
                    "  Tile (%d,%d): +%d POI (total: %d)",
                    tx, ty, len(filtered_pois), len(self._results),
                )

            # ── Expand hàng xóm vào queue ──
            added_neighbors = 0
            for n_tile in neighbors:
                if (
                    n_tile in self._all_tiles_set
                    and n_tile not in self._visited
                    and n_tile not in self._discarded
                    and n_tile not in self._queued
                ):
                    # Thêm hàng xóm nếu tile hiện tại có POI,
                    # hoặc EXPAND_EMPTY=True, hoặc geo_inside=True (đảm bảo không bỏ sót)
                    if filtered_pois or EXPAND_EMPTY or geo_inside:
                        await self._queue.put(n_tile)
                        self._queued.add(n_tile)
                        added_neighbors += 1

            if added_neighbors:
                logger.debug(
                    "  Tile (%d,%d): queued %d neighbors",
                    tx, ty, added_neighbors,
                )

            # ── Lưu checkpoint và cập nhật map ──
            await self._save_checkpoint()
            await self._save_results()
            self._update_map()

    # ── Properties ───────────────────────────────────────────

    @property
    def is_done(self) -> bool:
        """True nếu queue rỗng và không còn tile hợp lệ chưa xử lý."""
        remaining = self._all_tiles_set - self._visited - self._discarded
        return self._queue.empty() and not remaining

    @property
    def stats(self) -> Dict:
        total = len(self._all_tiles)
        done = len(self._visited)
        disc = len(self._discarded)
        return {
            "total_tiles":  total,
            "tiles_done":   done,
            "tiles_disc":   disc,
            "pct_done":     done / total * 100 if total else 0.0,
            "pois_found":   len(self._results),
            "queue_size":   self._queue.qsize(),
        }

    # ── Checkpoint I/O ───────────────────────────────────────

    def _load_checkpoint(self) -> Tuple[Set[TileCoord], Set[TileCoord]]:
        """Nạp checkpoint.json, trả về (visited_set, queued_set)."""
        if not os.path.exists(CHECKPOINT_FILE):
            return set(), set()
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            visited = {tuple(t) for t in data.get("visited", [])}
            queued  = {tuple(t) for t in data.get("queue",   [])}
            logger.info(
                "Checkpoint loaded: %d visited, %d queued",
                len(visited), len(queued),
            )
            return visited, queued
        except Exception as exc:
            logger.warning("Could not load checkpoint: %s", exc)
            return set(), set()

    def _load_discarded(self) -> Set[TileCoord]:
        """Nạp danh sách tile discarded từ checkpoint."""
        if not os.path.exists(CHECKPOINT_FILE):
            return set()
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {tuple(t) for t in data.get("discarded", [])}
        except Exception:
            return set()

    async def _save_checkpoint(self) -> None:
        """Ghi trạng thái hiện tại ra checkpoint.json (dùng tmp file để an toàn)."""
        tmp = CHECKPOINT_FILE + ".tmp"
        try:
            data = {
                "visited":   [list(t) for t in self._visited],
                "queue":     [list(t) for t in self._queued - self._visited],
                "discarded": [list(t) for t in self._discarded],
            }
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, CHECKPOINT_FILE)
        except Exception as exc:
            logger.warning("Could not save checkpoint: %s", exc)

    # ── Results I/O ──────────────────────────────────────────

    def _load_results(self) -> List[dict]:
        """Nạp results.json nếu tồn tại."""
        if not os.path.exists(RESULTS_FILE):
            return []
        try:
            with open(RESULTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info("Loaded %d existing POIs from results.json", len(data))
            return data
        except Exception as exc:
            logger.warning("Could not load results: %s", exc)
            return []

    async def _save_results(self) -> None:
        """Ghi kết quả POI ra results.json (dùng tmp file để an toàn)."""
        tmp = RESULTS_FILE + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._results, f, ensure_ascii=False, indent=2)
            os.replace(tmp, RESULTS_FILE)
        except Exception as exc:
            logger.warning("Could not save results: %s", exc)

    # ── Map viewer update ────────────────────────────────────

    def _update_map(self) -> None:
        """Tái sinh map_viewer.html với trạng thái mới nhất."""
        try:
            from map_viewer import build_and_save
            build_and_save(
                self._all_tiles,
                self._visited,
                self._queued,
                len(self._results),
                self._discarded,
            )
        except Exception as exc:
            logger.debug("Could not update map viewer: %s", exc)

    # ── Internal helpers ─────────────────────────────────────

    def _find_frontier_tiles(self) -> Set[TileCoord]:
        """
        Tìm các tile chưa xử lý kề với các tile đã xử lý (frontier BFS).
        Dùng để tiếp tục scan khi queue trống nhưng vẫn còn tile chưa quét.
        """
        frontier: Set[TileCoord] = set()
        for tx, ty in self._visited:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    n = (tx + dx, ty + dy)
                    if (
                        n in self._all_tiles_set
                        and n not in self._visited
                        and n not in self._discarded
                        and n not in self._queued
                    ):
                        frontier.add(n)
        return frontier


# ── Utilities ────────────────────────────────────────────────

def _sort_by_distance(
    tiles: Set[TileCoord],
    center_lat: float,
    center_lng: float,
) -> List[TileCoord]:
    """
    Sắp xếp danh sách tile theo khoảng cách Euclid từ tọa độ trung tâm.
    Tile gần nhất được xếp trước để đảm bảo quét outward từ trung tâm.
    """
    from grid import tile_center as _tile_center
    def _dist(tile: TileCoord) -> float:
        clat, clng = _tile_center(tile[0], tile[1], ZOOM_LEVEL)
        return (clat - center_lat) ** 2 + (clng - center_lng) ** 2

    return sorted(tiles, key=_dist)
