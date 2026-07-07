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
import csv
import json
import logging
import os
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import config
from config import (
    EXPAND_EMPTY,
    RADIUS_KM,
    ZOOM_LEVEL,
)
from grid import (
    generate_all_tiles,
    is_point_in_boundary,
    lat_lng_to_tile,
    load_boundary_from_path,
    load_or_download_boundary,
    tile_center,
)
from scan_context import ScanArea, get_single_config_area

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

    def __init__(self, area: Optional[ScanArea] = None) -> None:
        self.area = area or get_single_config_area()
        os.makedirs(self.area.runtime_dir, exist_ok=True)
        self._all_tiles: List[TileCoord] = []
        self._all_tiles_set: Set[TileCoord] = set()
        self._visited: Set[TileCoord] = set()
        self._discarded: Set[TileCoord] = set()
        self._queued: Set[TileCoord] = set()
        self._captured: Set[TileCoord] = set()
        self._tile_bboxes: Dict[TileCoord, Tuple[float, float, float, float]] = {}
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
        # 1. Tải ranh giới quận/huyện
        logger.info("Loading boundary for '%s, %s'...", self.area.district, self.area.province)
        if self.area.boundary_path:
            self._boundary = load_boundary_from_path(self.area.boundary_path)
        else:
            self._boundary = load_or_download_boundary(self.area.district, self.area.province)
        if self._boundary:
            logger.info("District boundary loaded successfully.")
        else:
            logger.warning(
                "Could not load boundary for '%s'. Geofencing disabled.",
                self.area.district,
            )

        # 2. Sinh danh sách tile
        self._all_tiles = generate_all_tiles(
            self.area.center_lat,
            self.area.center_lng,
            RADIUS_KM,
            ZOOM_LEVEL,
            self.area.district,
            boundary_geometry=self._boundary,
        )
        self._all_tiles_set = set(self._all_tiles)
        logger.info("Total tiles to scan: %d", len(self._all_tiles))

        # 3. Nạp checkpoint
        visited_from_checkpoint, queued_from_checkpoint = self._load_checkpoint()
        self._results = _deduplicate_pois(self._load_results())
        self._save_clean_results_csv_sync()

        # 4. Xác định tile ban đầu (tile chứa tâm khu vực)
        center_tile = lat_lng_to_tile(
            self.area.center_lat,
            self.area.center_lng,
            ZOOM_LEVEL,
            self.area.center_lat,
            self.area.center_lng,
        )
        if center_tile not in self._all_tiles_set:
            # Nếu tâm không trong danh sách, lấy tile gần tâm nhất
            if self._all_tiles:
                center_tile = _sort_by_distance(set(self._all_tiles), self.area.center_lat, self.area.center_lng)[0]

        # 5. Khởi tạo queue từ checkpoint nếu có, hoặc bắt đầu mới hoàn toàn
        if visited_from_checkpoint or queued_from_checkpoint:
            self._visited = visited_from_checkpoint & self._all_tiles_set
            self._discarded = self._load_discarded()

            # Đưa lại các tile đã queued từ checkpoint vào queue (ưu tiên xử lý trước)
            seed_tiles = queued_from_checkpoint - self._visited - self._discarded
            for tile in _sort_by_distance(seed_tiles, self.area.center_lat, self.area.center_lng):
                await self._queue.put(tile)
                self._queued.add(tile)

            # Đưa TOÀN BỘ tile còn lại chưa được quét vào queue
            # → đảm bảo không bỏ sót tile cô lập
            # → sắp xếp lan tỏa từ tọa độ xuất phát ra ngoài
            remaining_unqueued = (
                self._all_tiles_set - self._visited - self._discarded - self._queued
            )
            if remaining_unqueued:
                logger.info(
                    "Found %d unscanned tiles not in checkpoint queue — re-queuing them.",
                    len(remaining_unqueued),
                )
                for tile in _sort_by_distance(remaining_unqueued, self.area.center_lat, self.area.center_lng):
                    await self._queue.put(tile)
                    self._queued.add(tile)

            logger.info(
                "Resumed from checkpoint: %d done, %d discarded, %d in queue",
                len(self._visited), len(self._discarded), self._queue.qsize(),
            )
        else:
            # Bắt đầu mới hoàn toàn: xếp toàn bộ tile theo thứ tự khoảng cách từ tọa độ xuất phát ra ngoài
            # Điều này giúp lan tỏa tròn đều từ tâm, ưu tiên Up, Down, Left, Right trước do khoảng cách nhỏ hơn góc chéo
            sorted_tiles = _sort_by_distance(self._all_tiles_set, self.area.center_lat, self.area.center_lng)
            for tile in sorted_tiles:
                await self._queue.put(tile)
                self._queued.add(tile)
            logger.info(
                "Starting fresh scan radiating outward from center coordinates. Total: %d tiles queued",
                len(self._all_tiles),
            )

        # 6. Khởi tạo bản đồ và map_status.json ngay tại vạch xuất phát để HTML kết nối realtime lập tức
        self._update_map()

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
            if self._queue.empty():
                remaining = self._all_tiles_set - self._visited - self._discarded - self._queued
                if remaining:
                    # Lấy tile tiếp theo gần tọa độ xuất phát (tâm) nhất
                    next_tile = _sort_by_distance(remaining, self.area.center_lat, self.area.center_lng)[0]
                    await self._queue.put(next_tile)
                    self._queued.add(next_tile)

        try:
            tile = self._queue.get_nowait()
            return tile
        except asyncio.QueueEmpty:
            return None

    async def report_captured(self, tile: TileCoord, bbox: Optional[Tuple[float, float, float, float]] = None) -> None:
        """Báo cáo rằng tile đã được chụp ảnh xong, đang gửi sang LLM."""
        async with self._lock:
            self._captured.add(tile)
            if bbox:
                self._tile_bboxes[tile] = bbox
            await self._save_checkpoint()
            self._update_map()

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
                clat, clng = tile_center(tx, ty, ZOOM_LEVEL, self.area.center_lat, self.area.center_lng)
                geo_inside = is_point_in_boundary(clat, clng, self._boundary, buffer_meters=100.0)

            if outside_district and geo_inside:
                logger.info(
                    "  [GeoOverride] LLM said outside but tile (%d,%d) IS inside polygon (buffered 100m) → overriding!",
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
                    # Cho phép lưu POI ở sát biên quận trong vòng 100m để tránh mất thông tin địa điểm ở vùng giáp ranh
                    if is_point_in_boundary(poi_lat, poi_lng, self._boundary, buffer_meters=100.0):
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
            self._results = _deduplicate_pois(self._results)

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

    @property
    def checkpoint_file(self) -> str:
        return self.area.checkpoint_file

    @property
    def results_file(self) -> str:
        return self.area.results_file

    # ── Checkpoint I/O ───────────────────────────────────────

    def _load_checkpoint(self) -> Tuple[Set[TileCoord], Set[TileCoord]]:
        """Nạp checkpoint.json, trả về (visited_set, queued_set)."""
        if not os.path.exists(self.checkpoint_file):
            return set(), set()
        try:
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            visited = {tuple(t) for t in data.get("visited", [])}
            queued  = {tuple(t) for t in data.get("queue",   [])}
            self._captured = {tuple(t) for t in data.get("captured", [])}
            
            self._tile_bboxes = {}
            for k, v in data.get("tile_bboxes", {}).items():
                try:
                    tx, ty = map(int, k.split(","))
                    self._tile_bboxes[(tx, ty)] = tuple(v)
                except Exception:
                    pass

            logger.info(
                "Checkpoint loaded: %d visited, %d queued, %d captured, %d custom bboxes",
                len(visited), len(queued), len(self._captured), len(self._tile_bboxes),
            )
            return visited, queued
        except Exception as exc:
            logger.warning("Could not load checkpoint: %s", exc)
            return set(), set()

    def _load_discarded(self) -> Set[TileCoord]:
        """Nạp danh sách tile discarded từ checkpoint."""
        if not os.path.exists(self.checkpoint_file):
            return set()
        try:
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {tuple(t) for t in data.get("discarded", [])}
        except Exception:
            return set()

    async def _save_checkpoint(self) -> None:
        """Ghi trạng thái hiện tại ra checkpoint.json (dùng tmp file để an toàn)."""
        tmp = self.checkpoint_file + ".tmp"
        try:
            data = {
                "visited":   [list(t) for t in self._visited],
                "queue":     [list(t) for t in self._queued - self._visited],
                "discarded": [list(t) for t in self._discarded],
                "captured":  [list(t) for t in self._captured],
                "tile_bboxes": {f"{t[0]},{t[1]}": list(bbox) for t, bbox in self._tile_bboxes.items()},
            }
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            _robust_replace(tmp, self.checkpoint_file)
        except Exception as exc:
            logger.warning("Could not save checkpoint: %s", exc)

    # ── Results I/O ──────────────────────────────────────────

    def _load_results(self) -> List[dict]:
        """Nạp results.json nếu tồn tại."""
        if not os.path.exists(self.results_file):
            return []
        try:
            with open(self.results_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info("Loaded %d existing POIs from results.json", len(data))
            return data
        except Exception as exc:
            logger.warning("Could not load results: %s", exc)
            return []

    async def _save_results(self) -> None:
        """Ghi kết quả POI ra results.json (dùng tmp file để an toàn)."""
        tmp = self.results_file + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._results, f, ensure_ascii=False, indent=2)
            _robust_replace(tmp, self.results_file)
            self._save_clean_results_csv_sync()
        except Exception as exc:
            logger.warning("Could not save results: %s", exc)

    def _save_clean_results_csv_sync(self) -> None:
        """Ghi file CSV sạch cuối cùng cho POI đã dedupe."""
        tmp = self.area.clean_csv_file + ".tmp"
        fieldnames = [
            "title_poi",
            "ten_dia_diem",
            "toa_do",
            "quan_huyen_xa",
            "tinh_thanh_pho",
            "crop_image",
            "tile_x",
            "tile_y",
        ]
        try:
            with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for poi in self._results:
                    name = (poi.get("name") or "").strip()
                    lat = poi.get("approx_lat")
                    lng = poi.get("approx_lng")
                    if not name:
                        continue
                    writer.writerow({
                        "title_poi": name,
                        "ten_dia_diem": name,
                        "toa_do": _format_coordinate(lat, lng),
                        "quan_huyen_xa": _format_admin_area(self.area),
                        "tinh_thanh_pho": self.area.province,
                        "crop_image": _format_crop_path(poi.get("crop_image") or poi.get("crop_path")),
                        "tile_x": poi.get("tile_x", ""),
                        "tile_y": poi.get("tile_y", ""),
                    })
            _robust_replace(tmp, self.area.clean_csv_file)
        except Exception as exc:
            logger.warning("Could not save clean CSV results: %s", exc)

    # ── Map viewer update ────────────────────────────────────

    def _update_map(self) -> None:
        """Tái sinh map_viewer.html với trạng thái mới nhất, và cập nhật map_status.json."""
        try:
            from map_viewer import build_and_save
            build_and_save(
                self._all_tiles,
                self._visited,
                self._queued,
                self._results,
                self._discarded,
                self._captured,
                out_path=self.area.map_viewer_file,
                tile_bboxes=self._tile_bboxes,
                center_lat=self.area.center_lat,
                center_lng=self.area.center_lng,
                target_district=self.area.district,
                boundary_geometry=self._boundary,
            )
        except Exception as exc:
            logger.debug("Could not update map viewer: %s", exc)

        # Ghi map_status.json → HTML sẽ poll file này để biết khi nào cần reload
        try:
            import time as _time
            status_path = self.area.status_file
            data = {
                "ts": _time.time(),
                "done": len(self._visited),
                "captured": len(self._captured),
                "pois": len(self._results),
            }
            tmp = status_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            _robust_replace(tmp, status_path)
        except Exception as exc:
            logger.debug("Could not write map_status.json: %s", exc)


# ── Utilities ────────────────────────────────────────────────

def _format_float(value: object) -> str:
    """Format số tọa độ ổn định cho CSV."""
    if value is None:
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return ""


def _format_coordinate(lat: object, lng: object) -> str:
    """Format cặp tọa độ lat,lng cho CSV."""
    lat_s = _format_float(lat)
    lng_s = _format_float(lng)
    if not lat_s or not lng_s:
        return ""
    return f"{lat_s}, {lng_s}"


def _format_admin_area(area: ScanArea) -> str:
    """Ghép xã/phường nếu có, rồi quận/huyện đang quét."""
    ward = (area.ward or "").strip()
    district = (area.district or "").strip()
    return ", ".join(part for part in (ward, district) if part)


def _format_crop_path(path_value: object) -> str:
    """Format crop path ngắn, dễ mở từ CSV."""
    if not path_value:
        return ""
    try:
        path = os.fspath(path_value)
    except TypeError:
        return ""
    try:
        rel = os.path.relpath(path, config.BASE_DIR)
    except Exception:
        rel = path
    return rel.replace("\\", "/")


def _sort_by_distance(
    tiles: Set[TileCoord],
    center_lat: float,
    center_lng: float,
) -> List[TileCoord]:
    """
    Sắp xếp danh sách tile theo khoảng cách hình học trong lưới custom (tx, ty)
    để đảm bảo lan tỏa tròn đều đối xứng, ưu tiên theo thứ tự: Lên, Xuống, Trái, Phải.
    """
    def _sort_key(tile: TileCoord) -> Tuple[float, float, float, float]:
        tx, ty = tile
        # 1. Khoảng cách Euclid bình phương trong không gian lưới
        grid_dist = tx ** 2 + ty ** 2
        # 2. Ưu tiên hướng Lên (0, -1), Xuống (0, 1) trước Trái/Phải (abs(tx)=0 < abs(tx)=1)
        abs_tx = abs(tx)
        # 3. Hướng Lên (ty < 0) trước Xuống (ty > 0)
        # 4. Hướng Trái (tx < 0) trước Phải (tx > 0)
        return (grid_dist, abs_tx, ty, tx)

    return sorted(tiles, key=_sort_key)


def _robust_replace(src: str, dst: str, max_retries: int = 5, delay: float = 0.05) -> None:
    """
    Thay thế file an toàn với retry tự động để chống lỗi khóa file (WinError 5) trên Windows.
    """
    import time
    for i in range(max_retries):
        try:
            os.replace(src, dst)
            return
        except OSError:
            if i == max_retries - 1:
                raise
            time.sleep(delay)


def _deduplicate_pois(pois: List[dict]) -> List[dict]:
    """
    Loại bỏ các POI trùng lặp dựa trên khoảng cách địa lý và độ tương đồng tên.
    Fix 6: Sử dụng similarity ratio thay vì substring đơn giản để tránh gộp nhầm.
    """
    from src.vision import _strip_vietnamese_accents
    import math
    
    def clean_name(name: str) -> str:
        s = _strip_vietnamese_accents(name)
        # Giữ lại các chữ cái và chữ số
        return "".join(c for c in s if c.isalnum())

    def name_similarity(a: str, b: str) -> float:
        """Tính tỷ lệ tương đồng chuỗi giữa 2 tên bằng SequenceMatcher."""
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0
        import difflib
        return difflib.SequenceMatcher(None, a, b).ratio()

    # Sắp xếp các POI theo chiều dài tên giảm dần để ưu tiên giữ tên đầy đủ hơn
    sorted_pois = sorted(pois, key=lambda p: len(p.get("name", "")), reverse=True)
    
    unique_pois = []
    for poi in sorted_pois:
        lat = poi.get("approx_lat")
        lng = poi.get("approx_lng")
        name = poi.get("name", "").strip()
        
        if lat is None or lng is None or not name:
            unique_pois.append(poi)
            continue
            
        c_name = clean_name(name)
        
        is_dup = False
        for upoi in unique_pois:
            u_lat = upoi.get("approx_lat")
            u_lng = upoi.get("approx_lng")
            u_name = upoi.get("name", "").strip()
            
            if u_lat is None or u_lng is None or not u_name:
                continue
                
            # Tính khoảng cách địa lý Haversine (mét)
            phi1 = math.radians(lat)
            phi2 = math.radians(u_lat)
            dphi = math.radians(u_lat - lat)
            dlng = math.radians(u_lng - lng)
            a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlng / 2) ** 2
            dist = 6371000.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            
            # Ngưỡng khoảng cách trùng lặp là 12 mét
            if dist < 12.0:
                uc_name = clean_name(u_name)
                
                # Trùng khớp hoàn toàn
                if c_name == uc_name:
                    is_dup = True
                    break
                
                # Kiểm tra tỷ lệ tương đồng >= 60% (tránh gộp nhầm POI khác nhau cùng khu vực)
                sim = name_similarity(c_name, uc_name)
                if sim >= 0.6:
                    is_dup = True
                    break
                    
        if not is_dup:
            unique_pois.append(poi)
            
    return unique_pois
