from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import config
from boundary_manager import read_manifest, slugify
from center_manager import get_center, read_centers


@dataclass(frozen=True)
class ScanArea:
    province: str
    district: str
    ward: str = ""
    center_lat: float = 0.0
    center_lng: float = 0.0
    boundary_path: str = ""
    runtime_dir: str = ""

    @property
    def checkpoint_file(self) -> str:
        return str(Path(self.runtime_dir) / "checkpoint.json")

    @property
    def results_file(self) -> str:
        return str(Path(self.runtime_dir) / "results.json")

    @property
    def clean_csv_file(self) -> str:
        return str(Path(self.runtime_dir) / "clean_results.csv")

    @property
    def status_file(self) -> str:
        return str(Path(self.runtime_dir) / "map_status.json")

    @property
    def map_viewer_file(self) -> str:
        return str(Path(self.runtime_dir) / "map_viewer.html")


def _runtime_dir_for(province: str, district: str) -> str:
    root = Path(getattr(config, "AREA_RUNTIME_ROOT", config.RUNTIME_DIR / "areas"))
    return str(root / slugify(province) / slugify(district))


_PROVINCE_PRIORITY = ["Thành phố Hồ Chí Minh"]
_DISTRICT_TYPE_ORDER = {
    "Quận": 0,
    "Quan": 0,
    "Thành phố": 1,
    "Thanh pho": 1,
    "Thị xã": 2,
    "Thi xa": 2,
    "Huyện": 3,
    "Huyen": 3,
}


def _strip_prefix(name: str) -> str:
    value = " ".join((name or "").split())
    for prefix in ("Thành phố ", "Thanh pho ", "Tỉnh ", "Tinh ", "Quận ", "Quan ", "Huyện ", "Huyen ", "Thị xã ", "Thi xa "):
        if value.startswith(prefix):
            return value[len(prefix):]
    return value


def _district_type_rank(name: str) -> int:
    for prefix, rank in _DISTRICT_TYPE_ORDER.items():
        if (name or "").startswith(prefix + " "):
            return rank
    return 9


def _district_number_rank(name: str) -> tuple[int, int | str]:
    simple = _strip_prefix(name)
    try:
        return (0, int(simple))
    except ValueError:
        return (1, simple.casefold())


def _province_rank(province: str) -> tuple[int, str]:
    for idx, priority in enumerate(_PROVINCE_PRIORITY):
        if province == priority:
            return (idx, "")
    return (len(_PROVINCE_PRIORITY), _strip_prefix(province).casefold())


def _manifest_sort_key(row: dict) -> tuple:
    province = str(row.get("province") or "")
    district = str(row.get("district") or "")
    return (
        _province_rank(province),
        _district_type_rank(district),
        _district_number_rank(district),
        _strip_prefix(district).casefold(),
    )


def get_single_config_area() -> ScanArea:
    return ScanArea(
        province=getattr(config, "TARGET_PROVINCE", ""),
        district=getattr(config, "TARGET_DISTRICT", ""),
        ward=getattr(config, "TARGET_WARD", ""),
        center_lat=float(getattr(config, "CENTER_LAT")),
        center_lng=float(getattr(config, "CENTER_LNG")),
        boundary_path="",
        runtime_dir=str(config.RUNTIME_DIR),
    )


def area_from_manifest_row(row: dict, centers: dict[str, dict] | None = None) -> Optional[ScanArea]:
    province = str(row.get("province") or "").strip()
    district = str(row.get("district") or "").strip()
    if not province or not district:
        return None
    centroid = row.get("centroid") or []
    center = get_center(province, district, centers)
    try:
        if center:
            center_lat = float(center["center_lat"])
            center_lng = float(center["center_lng"])
        else:
            center_lat = float(centroid[0])
            center_lng = float(centroid[1])
    except Exception:
        return None
    boundary_path = str(row.get("boundary_path") or "")
    if boundary_path and not Path(boundary_path).is_absolute():
        boundary_path = str(config.BASE_DIR / boundary_path)
    return ScanArea(
        province=province,
        district=district,
        ward=str(row.get("ward") or ""),
        center_lat=center_lat,
        center_lng=center_lng,
        boundary_path=boundary_path,
        runtime_dir=_runtime_dir_for(province, district),
    )


def iter_manifest_areas(
    province_filter: str = "",
    district_filter: str = "",
    start_index: int = 0,
    max_areas: int = 0,
) -> Iterator[ScanArea]:
    province_filter = (province_filter or "").casefold().strip()
    district_filter = (district_filter or "").casefold().strip()
    centers = read_centers()
    emitted = 0
    for idx, row in enumerate(sorted(read_manifest(), key=_manifest_sort_key)):
        if idx < start_index:
            continue
        area = area_from_manifest_row(row, centers)
        if area is None:
            continue
        if province_filter and province_filter not in area.province.casefold():
            continue
        if district_filter and district_filter not in area.district.casefold():
            continue
        yield area
        emitted += 1
        if max_areas and emitted >= max_areas:
            break
